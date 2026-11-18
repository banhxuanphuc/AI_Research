import asyncio

import httpx
from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from research_scientist.core.config import get_settings
from research_scientist.core.db import get_session
from research_scientist.ingestion.storage import ObjectStore
from research_scientist.retrieval.vector_store import ChunkVectorStore

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness: process còn sống."""
    return {"status": "ok"}


async def _http_ok(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            return (await client.get(url)).status_code < 500
    except httpx.HTTPError:
        return False


@router.get("/health/ready")
async def ready(request: Request, session: AsyncSession = Depends(get_session)) -> dict:
    """Readiness: kiểm tra từng dependency. Service profile 'ai' là tuỳ chọn."""
    s = get_settings()

    async def db() -> bool:
        try:
            await session.execute(text("select 1"))
            return True
        except Exception:
            return False

    async def redis() -> bool:
        try:
            return bool(await request.app.state.arq.ping())
        except Exception:
            return False

    async def minio() -> bool:
        try:
            return await asyncio.to_thread(ObjectStore().bucket_ready)
        except Exception:
            return False

    store = ChunkVectorStore(s.qdrant_url, s.qdrant_collection, s.embed_dim)
    names = ["postgres", "redis", "qdrant", "minio", "grobid", "tei_embed"]
    results = await asyncio.gather(
        db(),
        redis(),
        store.health(),
        minio(),
        _http_ok(f"{s.grobid_url}/api/isalive"),
        _http_ok(f"{s.tei_embed_url}/health"),
    )
    checks = dict(zip(names, results, strict=True))
    core_ok = all(checks[n] for n in ("postgres", "redis", "qdrant", "minio"))
    return {
        "status": "ready" if core_ok else "degraded",
        "llm_default_model": s.llm_default_model,
        "anthropic_key_configured": s.anthropic_key() is not None,
        "gemini_key_configured": s.gemini_key() is not None,
        "checks": checks,
    }
