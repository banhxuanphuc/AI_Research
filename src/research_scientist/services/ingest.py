"""Use-case ingestion: PDF trong MinIO -> parse -> sections/chunks (Postgres) -> vectors (Qdrant).

Chạy trong arq worker. Idempotent theo paper_version: chạy lại sẽ xoá section/chunk cũ.
"""

import asyncio
import logging
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from research_scientist.core.config import get_settings
from research_scientist.domain.models import Chunk, Paper, PaperSection, PaperVersion
from research_scientist.ingestion.chunking import CHUNKER_VERSION, chunk_sections
from research_scientist.ingestion.grobid import GrobidClient, ParsedDocument, parse_tei
from research_scientist.ingestion.pdf_fallback import parse_pdf_pages
from research_scientist.ingestion.storage import ObjectStore
from research_scientist.providers.embeddings import TEIEmbedder
from research_scientist.retrieval.vector_store import ChunkVectorStore

log = logging.getLogger(__name__)


async def parse_pdf(pdf: bytes) -> ParsedDocument:
    s = get_settings()
    grobid = GrobidClient(s.grobid_url)
    version = await grobid.version()
    if version:
        try:
            return parse_tei(await grobid.process_fulltext(pdf), parser_version=version)
        except Exception:
            log.exception("GROBID failed, falling back to PyMuPDF")
    return parse_pdf_pages(pdf)


async def ingest_paper_version(
    session: AsyncSession, version_id: uuid.UUID, workspace_id: uuid.UUID
) -> dict:
    s = get_settings()
    version = await session.get(PaperVersion, version_id)
    if version is None:
        raise ValueError(f"paper_version {version_id} not found")
    paper = await session.get(Paper, version.paper_id)
    assert paper is not None

    key = version.file_uri.split("/", 3)[3]  # s3://bucket/<key>
    pdf = await asyncio.to_thread(ObjectStore().get, key)
    doc = await parse_pdf(pdf)

    # Idempotent: xoá dữ liệu parse cũ của version này
    old_sections = select(PaperSection.id).where(PaperSection.paper_version_id == version.id)
    await session.execute(delete(Chunk).where(Chunk.section_id.in_(old_sections)))
    await session.execute(delete(PaperSection).where(PaperSection.paper_version_id == version.id))

    section_rows = [
        PaperSection(
            paper_version_id=version.id,
            ordinal=i,
            section_path=sec.path,
            page_start=next((p.page for p in sec.paragraphs if p.page), None),
            page_end=next((p.page for p in reversed(sec.paragraphs) if p.page), None),
            text=sec.text,
        )
        for i, sec in enumerate(doc.sections)
    ]
    session.add_all(section_rows)
    await session.flush()

    drafts = chunk_sections(doc.sections)
    chunk_rows = [
        Chunk(
            workspace_id=workspace_id,
            paper_id=paper.id,
            section_id=section_rows[d.section_index].id,
            ordinal=d.ordinal,
            section_path=d.section_path,
            page=d.page,
            char_start=d.char_start,
            char_end=d.char_end,
            text=d.text,
            chunker_version=CHUNKER_VERSION,
        )
        for d in drafts
    ]
    session.add_all(chunk_rows)

    version.parser = doc.parser
    version.parser_version = doc.parser_version
    if doc.title and paper.title.lower().endswith(".pdf"):
        paper.title = doc.title  # thay tên file tạm bằng title parse được
    await session.flush()

    embedded = 0
    embedder = TEIEmbedder(s.tei_embed_url)
    if chunk_rows and await embedder.health():
        vectors = await embedder.embed([d.embed_text for d in drafts])
        store = ChunkVectorStore(s.qdrant_url, s.qdrant_collection, s.embed_dim)
        await store.ensure_collection()
        await store.upsert(
            ids=[str(c.id) for c in chunk_rows],
            vectors=vectors,
            payloads=[
                {
                    "workspace_id": str(workspace_id),
                    "paper_id": str(paper.id),
                    "doi": paper.doi,
                    "section": c.section_path,
                    "page": c.page,
                    "year": paper.year,
                    "chunker_version": CHUNKER_VERSION,
                }
                for c in chunk_rows
            ],
        )
        embedded = len(chunk_rows)
    else:
        log.warning("TEI embedder unavailable - chunks stored without vectors")

    await session.commit()
    return {
        "parser": doc.parser,
        "sections": len(section_rows),
        "chunks": len(chunk_rows),
        "embedded": embedded,
        "references": doc.reference_count,
    }
