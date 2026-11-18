"""Client cho các nguồn học thuật: Semantic Scholar, arXiv, Crossref.

- Tôn trọng rate limit: retry có backoff với 429/5xx, timeout rõ ràng.
- Chỉ gọi domain trong allowlist (xem security/policy.py).
- Không đồng nhất paperId của Semantic Scholar với DOI.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
from lxml import etree
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from research_scientist.ingestion.dedup import normalize_arxiv_id, normalize_doi
from research_scientist.schemas.research import PaperCandidate

USER_AGENT = "ai-research-scientist/0.1 (research prototype)"
S2_FIELDS = "title,externalIds,authors,year,venue,abstract,url,openAccessPdf,publicationTypes"


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError)


_retry = retry(
    retry=retry_if_exception(_retryable),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)


class SemanticScholarClient:
    BASE = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str | None = None, timeout: float = 20.0):
        headers = {"User-Agent": USER_AGENT}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.AsyncClient(base_url=self.BASE, headers=headers, timeout=timeout)

    @_retry
    async def search(
        self, query: str, limit: int = 20, year_from: int | None = None, year_to: int | None = None
    ) -> list[PaperCandidate]:
        params: dict[str, Any] = {"query": query, "limit": limit, "fields": S2_FIELDS}
        if year_from or year_to:
            params["year"] = f"{year_from or ''}-{year_to or ''}"
        resp = await self._client.get("/paper/search", params=params)
        resp.raise_for_status()
        return [self._to_candidate(p) for p in resp.json().get("data", [])]

    @staticmethod
    def _to_candidate(p: dict[str, Any]) -> PaperCandidate:
        ext = p.get("externalIds") or {}
        types = p.get("publicationTypes") or []
        oa = p.get("openAccessPdf") or {}
        return PaperCandidate(
            title=p.get("title") or "",
            doi=normalize_doi(ext.get("DOI")),
            s2_id=p.get("paperId"),
            arxiv_id=normalize_arxiv_id(ext.get("ArXiv")),
            authors=[a.get("name", "") for a in p.get("authors") or []],
            year=p.get("year"),
            venue=p.get("venue") or None,
            abstract=p.get("abstract"),
            url=p.get("url"),
            open_access_pdf=oa.get("url") or None,
            is_preprint=bool(ext.get("ArXiv")) and not ext.get("DOI") and "JournalArticle" not in types,
            sources=["semantic_scholar"],
        )


class ArxivClient:
    BASE = "https://export.arxiv.org/api/query"
    NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def __init__(self, timeout: float = 30.0):
        self._client = httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=timeout)

    @_retry
    async def search(self, query: str, limit: int = 20) -> list[PaperCandidate]:
        params = {"search_query": f"all:{query}", "start": 0, "max_results": limit}
        resp = await self._client.get(self.BASE, params=params)
        resp.raise_for_status()
        return self.parse_feed(resp.content)

    @classmethod
    def parse_feed(cls, xml: bytes) -> list[PaperCandidate]:
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        root = etree.fromstring(xml, parser=parser)
        out: list[PaperCandidate] = []
        for entry in root.findall("a:entry", cls.NS):
            title = " ".join((entry.findtext("a:title", "", cls.NS) or "").split())
            published = entry.findtext("a:published", "", cls.NS) or ""
            pdf = next(
                (
                    link.get("href")
                    for link in entry.findall("a:link", cls.NS)
                    if link.get("title") == "pdf"
                ),
                None,
            )
            doi = entry.findtext("arxiv:doi", None, cls.NS)
            out.append(
                PaperCandidate(
                    title=title,
                    doi=normalize_doi(doi),
                    arxiv_id=normalize_arxiv_id(entry.findtext("a:id", "", cls.NS)),
                    authors=[
                        a.findtext("a:name", "", cls.NS) for a in entry.findall("a:author", cls.NS)
                    ],
                    year=int(published[:4]) if published[:4].isdigit() else None,
                    abstract=" ".join((entry.findtext("a:summary", "", cls.NS) or "").split()),
                    url=entry.findtext("a:id", None, cls.NS),
                    open_access_pdf=pdf,
                    is_preprint=doi is None,
                    sources=["arxiv"],
                )
            )
        return out


class CrossrefClient:
    BASE = "https://api.crossref.org"

    def __init__(self, mailto: str | None = None, timeout: float = 20.0):
        self._mailto = mailto
        self._client = httpx.AsyncClient(
            base_url=self.BASE, headers={"User-Agent": USER_AGENT}, timeout=timeout
        )

    @_retry
    async def get_work(self, doi: str) -> dict[str, Any] | None:
        """Tra cứu metadata canonical theo DOI (dùng cho citation verification)."""
        params = {"mailto": self._mailto} if self._mailto else None
        resp = await self._client.get(f"/works/{doi}", params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        msg = resp.json()["message"]
        issued = (msg.get("issued") or {}).get("date-parts") or [[None]]
        return {
            "doi": normalize_doi(msg.get("DOI")),
            "title": (msg.get("title") or [""])[0],
            "authors": [
                f"{a.get('given', '')} {a.get('family', '')}".strip() for a in msg.get("author", [])
            ],
            "year": issued[0][0],
            "venue": (msg.get("container-title") or [None])[0],
            "type": msg.get("type"),
        }


@dataclass
class DiscoveryResult:
    candidates: list[PaperCandidate]
    errors: list[str] = field(default_factory=list)  # "<nguồn> '<query>': <lỗi>"


def _describe(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


async def discover(
    queries: list[str],
    s2: SemanticScholarClient,
    arxiv: ArxivClient,
    limit_per_query: int = 15,
    year_from: int | None = None,
    year_to: int | None = None,
) -> DiscoveryResult:
    """Gọi song song các nguồn cho mọi query; lỗi của một nguồn không làm hỏng cả lượt,
    nhưng được trả về trong `errors` để người dùng biết kết quả có thể thiếu."""
    tasks, labels = [], []
    for q in queries:
        tasks.append(s2.search(q, limit_per_query, year_from, year_to))
        labels.append(f"semantic_scholar {q!r}")
        tasks.append(arxiv.search(q, limit_per_query))
        labels.append(f"arxiv {q!r}")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    candidates: list[PaperCandidate] = []
    errors: list[str] = []
    for label, res in zip(labels, results, strict=True):
        if isinstance(res, BaseException):
            errors.append(f"{label}: {_describe(res)}")
            continue
        candidates.extend(res)
    # arXiv API không lọc theo năm -> lọc sau
    kept = [
        c
        for c in candidates
        if c.year is None
        or ((year_from is None or c.year >= year_from) and (year_to is None or c.year <= year_to))
    ]
    return DiscoveryResult(candidates=kept, errors=errors)
