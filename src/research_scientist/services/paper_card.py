"""Use-case: extract a PaperCard for an ingested paper and store it with its evidence report.

Context engineering choices:
- Chunks are sent in reading order with short ids (c1, c2, ...) instead of UUIDs: fewer tokens
  and fewer copy errors by the model. Ids are mapped back to chunk UUIDs before storing.
- A character budget keeps very long papers from blowing up cost; the cut is reported.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from research_scientist.agents.paper_analyst import ChunkInput, extract_paper_card
from research_scientist.domain.models import Chunk, Paper, PaperCardRecord
from research_scientist.providers.gateway import LLMGateway
from research_scientist.schemas.paper_card import PAPER_CARD_SCHEMA_VERSION

log = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 240_000  # ~60k tokens


@dataclass
class CardContext:
    chunks: list[ChunkInput]
    id_map: dict[str, uuid.UUID]  # short id -> chunk UUID
    total_chunks: int
    truncated: bool


def build_context(rows: list[Chunk], max_chars: int = MAX_CONTEXT_CHARS) -> CardContext:
    chunks: list[ChunkInput] = []
    id_map: dict[str, uuid.UUID] = {}
    used = 0
    for row in rows:
        if used + len(row.text) > max_chars and chunks:
            break
        short = f"c{len(chunks) + 1}"
        chunks.append(ChunkInput(short, row.section_path, row.page, row.text))
        id_map[short] = row.id
        used += len(row.text)
    return CardContext(chunks, id_map, total_chunks=len(rows), truncated=len(chunks) < len(rows))


def remap_chunk_ids(obj: Any, id_map: dict[str, uuid.UUID]) -> Any:
    """Replace short ids with chunk UUIDs everywhere in the card JSON (unknown ids kept as-is)."""
    if isinstance(obj, dict):
        out = {k: remap_chunk_ids(v, id_map) for k, v in obj.items()}
        if isinstance(out.get("chunk_id"), str) and out["chunk_id"] in id_map:
            out["chunk_id"] = str(id_map[out["chunk_id"]])
        return out
    if isinstance(obj, list):
        return [remap_chunk_ids(v, id_map) for v in obj]
    return obj


async def extract_and_store_card(
    session: AsyncSession,
    gateway: LLMGateway,
    paper_id: uuid.UUID,
    workspace_id: uuid.UUID,
    model_override: str | None = None,
) -> PaperCardRecord:
    paper = await session.get(Paper, paper_id)
    if paper is None:
        raise ValueError(f"paper {paper_id} not found")
    rows = list(
        (
            await session.scalars(
                select(Chunk)
                .where(Chunk.paper_id == paper_id, Chunk.workspace_id == workspace_id)
                .order_by(Chunk.ordinal)
            )
        ).all()
    )
    if not rows:
        raise ValueError(f"paper {paper_id} has no chunks yet - ingest it first")

    ctx = build_context(rows)
    outcome = await extract_paper_card(gateway, paper.title, ctx.chunks, model_override)
    ev = outcome.evidence
    record = PaperCardRecord(
        paper_id=paper.id,
        schema_version=PAPER_CARD_SCHEMA_VERSION,
        prompt_version=outcome.prompt_id,
        model=outcome.llm.model,
        card=remap_chunk_ids(outcome.card.model_dump(), ctx.id_map),
        evidence_coverage=ev.key_field_coverage,
        evidence_report={
            "evidence_total": ev.total,
            "evidence_verified": ev.verified,
            "evidence_precision": round(ev.precision, 4),
            "key_field_coverage": round(ev.key_field_coverage, 4),
            "unknown_chunk": ev.unknown_chunk,
            "quote_mismatch": ev.quote_mismatch,
            "context_chunks": len(ctx.chunks),
            "total_chunks": ctx.total_chunks,
            "context_truncated": ctx.truncated,
        },
        input_tokens=outcome.llm.usage.input_tokens,
        output_tokens=outcome.llm.usage.output_tokens,
        cost_usd=outcome.llm.cost_usd,
        latency_ms=outcome.llm.latency_ms,
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    log.info("paper_card_stored", extra={"extra_fields": {"paper_id": str(paper.id), **record.evidence_report}})
    return record
