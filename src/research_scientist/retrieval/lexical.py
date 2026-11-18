"""Full-text search trên Postgres: bắt tên model, metric, acronym, DOI mà dense hay bỏ lỡ."""

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from research_scientist.domain.models import Chunk


async def lexical_search(
    session: AsyncSession, query: str, workspace_id: uuid.UUID, limit: int = 30
) -> list[tuple[str, float]]:
    tsquery = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank_cd(Chunk.tsv, tsquery).label("rank")
    stmt = (
        select(Chunk.id, rank)
        .where(Chunk.workspace_id == workspace_id, Chunk.tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [(str(r.id), float(r.rank)) for r in rows]
