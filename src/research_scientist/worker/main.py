"""arq worker: chạy các job dài (ingest PDF, embed, batch PaperCard).

Chạy: `arq research_scientist.worker.main.WorkerSettings`
"""

import uuid
from typing import Any

from arq.connections import RedisSettings

from research_scientist.core.config import get_settings
from research_scientist.core.db import get_sessionmaker
from research_scientist.core.logging import setup_logging
from research_scientist.services.ingest import ingest_paper_version


async def ingest_pdf(ctx: dict[str, Any], version_id: str, workspace_id: str) -> dict:
    async with get_sessionmaker()() as session:
        return await ingest_paper_version(session, uuid.UUID(version_id), uuid.UUID(workspace_id))


async def startup(ctx: dict[str, Any]) -> None:
    setup_logging()


class WorkerSettings:
    functions = [ingest_pdf]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 4
    job_timeout = 900  # giây; parse + embed một paper dài trên CPU
    max_tries = 3
