"""Register a PDF (uploaded or open-access) as a paper version and link it to a project.

Idempotent by identity: a paper with the same DOI or arXiv id is reused, and a version with the
same checksum is reused, so re-adding a paper never violates the unique DOI constraint.
"""

import asyncio
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from research_scientist.domain.models import (
    AccessStatus,
    Paper,
    PaperIdentifier,
    PaperVersion,
    ProjectPaper,
    ProjectPaperStatus,
)
from research_scientist.ingestion.dedup import normalize_doi, normalize_title
from research_scientist.ingestion.storage import ObjectStore, sha256
from research_scientist.schemas.research import PaperCandidate


@dataclass
class Registered:
    paper: Paper
    version: PaperVersion
    is_new_version: bool


async def _find_paper(session: AsyncSession, doi: str | None, arxiv_id: str | None) -> Paper | None:
    if doi:
        paper = await session.scalar(select(Paper).where(Paper.doi == doi))
        if paper:
            return paper
    if arxiv_id:
        return await session.scalar(
            select(Paper)
            .join(PaperIdentifier, PaperIdentifier.paper_id == Paper.id)
            .where(PaperIdentifier.scheme == "arxiv", PaperIdentifier.value == arxiv_id)
        )
    return None


async def register_pdf(
    session: AsyncSession,
    *,
    project_id: uuid.UUID,
    workspace_id: uuid.UUID,
    pdf: bytes,
    access_status: AccessStatus,
    title: str,
    meta: PaperCandidate | None = None,
    doi: str | None = None,
    license: str | None = None,
) -> Registered:
    doi = normalize_doi(doi or (meta.doi if meta else None))
    arxiv_id = meta.arxiv_id if meta else None

    paper = await _find_paper(session, doi, arxiv_id)
    if paper is None:
        paper = Paper(
            title=meta.title if meta else title,
            title_normalized=normalize_title(meta.title if meta else title),
            doi=doi,
            authors=meta.authors if meta else [],
            year=meta.year if meta else None,
            venue=meta.venue if meta else None,
            abstract=meta.abstract if meta else None,
            is_peer_reviewed=(not meta.is_preprint) if meta else None,
            source=(meta.sources[0] if meta and meta.sources else "user_upload"),
        )
        session.add(paper)
        await session.flush()
        if arxiv_id:
            session.add(PaperIdentifier(paper_id=paper.id, scheme="arxiv", value=arxiv_id))
        if meta and meta.s2_id:
            session.add(PaperIdentifier(paper_id=paper.id, scheme="s2", value=meta.s2_id))

    checksum = sha256(pdf)
    version = await session.scalar(
        select(PaperVersion).where(
            PaperVersion.paper_id == paper.id, PaperVersion.checksum_sha256 == checksum
        )
    )
    is_new = version is None
    if version is None:
        key = f"{workspace_id}/{paper.id}/{checksum}.pdf"
        uri = await asyncio.to_thread(ObjectStore().put, key, pdf, "application/pdf")
        version = PaperVersion(
            paper_id=paper.id,
            file_uri=uri,
            checksum_sha256=checksum,
            license=license,
            access_status=access_status,
        )
        session.add(version)

    link = await session.get(ProjectPaper, (project_id, paper.id))
    if link is None:
        session.add(
            ProjectPaper(
                project_id=project_id, paper_id=paper.id, status=ProjectPaperStatus.included
            )
        )
    await session.commit()
    return Registered(paper=paper, version=version, is_new_version=is_new)
