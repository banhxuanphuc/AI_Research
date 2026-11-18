import httpx

from research_scientist.ingestion.scholarly import discover
from research_scientist.schemas.research import PaperCandidate


class RateLimitedS2:
    async def search(self, q, limit, year_from, year_to):
        req = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search")
        raise httpx.HTTPStatusError("429", request=req, response=httpx.Response(429, request=req))


class FakeArxiv:
    async def search(self, q, limit):
        return [
            PaperCandidate(title="Old", year=2010, sources=["arxiv"]),
            PaperCandidate(title="New", year=2022, sources=["arxiv"]),
        ]


async def test_source_errors_are_reported_and_year_filter_applies():
    result = await discover(["q1"], RateLimitedS2(), FakeArxiv(), year_from=2019)
    assert [c.title for c in result.candidates] == ["New"]
    assert result.errors == ["semantic_scholar 'q1': HTTP 429"]
