"""Tìm paper trực tiếp (không qua LLM) - hữu ích để kiểm tra nguồn học thuật & dedup."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from research_scientist.core.config import get_settings
from research_scientist.ingestion.dedup import dedup_candidates
from research_scientist.ingestion.scholarly import ArxivClient, SemanticScholarClient, discover
from research_scientist.schemas.research import PaperCandidate

router = APIRouter(prefix="/discovery", tags=["discovery"])


class SearchRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=6)
    limit_per_query: int = Field(default=10, ge=1, le=50)
    year_from: int | None = None
    year_to: int | None = None


class SearchResponse(BaseModel):
    raw_count: int
    candidates: list[PaperCandidate]
    source_errors: list[str]


def scholarly_clients() -> tuple[SemanticScholarClient, ArxivClient]:
    s = get_settings()
    key = s.semantic_scholar_api_key.get_secret_value() if s.semantic_scholar_api_key else None
    return SemanticScholarClient(api_key=key or None), ArxivClient()


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest) -> SearchResponse:
    s2, arxiv = scholarly_clients()
    result = await discover(
        body.queries, s2, arxiv, body.limit_per_query, body.year_from, body.year_to
    )
    return SearchResponse(
        raw_count=len(result.candidates),
        candidates=dedup_candidates(result.candidates),
        source_errors=result.errors,
    )
