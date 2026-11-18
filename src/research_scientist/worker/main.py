"""arq worker: chạy các job dài (ingest PDF, embed, trích PaperCard).

Chạy: `arq research_scientist.worker.main.WorkerSettings`
"""

import uuid
from typing import Any

from arq.connections import RedisSettings

from research_scientist.core.config import get_settings
from research_scientist.core.db import get_sessionmaker
from research_scientist.core.logging import setup_logging
from research_scientist.domain.models import PaperVersion
from research_scientist.providers.gateway import build_gateway
from research_scientist.services.ingest import ingest_paper_version
from research_scientist.services.paper_card import extract_and_store_card


async def ingest_pdf(
    ctx: dict[str, Any], version_id: str, workspace_id: str, then_extract_card: bool = False
) -> dict:
    async with get_sessionmaker()() as session:
        result = await ingest_paper_version(session, uuid.UUID(version_id), uuid.UUID(workspace_id))
        if then_extract_card and result["chunks"]:
            version = await session.get(PaperVersion, uuid.UUID(version_id))
            assert version is not None
            job = await ctx["redis"].enqueue_job("extract_card", str(version.paper_id), workspace_id)
            result["card_job_id"] = job.job_id if job else None
        return result


async def extract_card(
    ctx: dict[str, Any], paper_id: str, workspace_id: str, model_override: str | None = None
) -> dict:
    gateway = build_gateway(get_settings().run_budget_usd)
    async with get_sessionmaker()() as session:
        record = await extract_and_store_card(
            session, gateway, uuid.UUID(paper_id), uuid.UUID(workspace_id), model_override
        )
        return {"card_id": str(record.id), "model": record.model, **record.evidence_report}


async def startup(ctx: dict[str, Any]) -> None:
    setup_logging()


class WorkerSettings:
    functions = [ingest_pdf, extract_card]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 4
    job_timeout = 900  # giây; parse + embed một paper dài trên CPU, hoặc một lượt LLM dài
    max_tries = 3
