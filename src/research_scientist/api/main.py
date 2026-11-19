"""FastAPI app.

Chạy local: `uvicorn research_scientist.api.main:app --reload`
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from research_scientist.api.routers import discovery, health, papers, projects, runs
from research_scientist.core.config import get_settings
from research_scientist.core.db import get_sessionmaker
from research_scientist.core.logging import setup_logging
from research_scientist.ingestion.storage import ObjectStore
from research_scientist.retrieval.vector_store import ChunkVectorStore
from research_scientist.services.workspace import ensure_default_workspace

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    s = get_settings()
    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(s.checkpoint_database_url)
        )
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        app.state.arq = await create_pool(RedisSettings.from_dsn(s.redis_url))
        stack.push_async_callback(app.state.arq.aclose)

        async with get_sessionmaker()() as session:
            await ensure_default_workspace(session)

        # Best-effort: hạ tầng phụ chưa sẵn sàng thì log cảnh báo, không chặn API khởi động
        try:
            await asyncio.to_thread(ObjectStore().ensure_bucket)
        except Exception:
            log.warning("MinIO not ready - bucket not ensured", exc_info=True)
        try:
            await ChunkVectorStore(
                s.qdrant_url, s.qdrant_collection, s.embed_dim
            ).ensure_collection()
        except Exception:
            log.warning("Qdrant not ready - collection not ensured", exc_info=True)
        yield


def create_app() -> FastAPI:
    app = FastAPI(title="AI Research Scientist API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(projects.router)
    app.include_router(papers.router)
    app.include_router(discovery.router)
    app.include_router(runs.router)
    return app


app = create_app()
