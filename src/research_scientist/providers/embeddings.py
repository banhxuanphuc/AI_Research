"""Client cho Hugging Face Text Embeddings Inference (TEI): embedding + rerank.

Anthropic không cung cấp embedding API, nên embedding/rerank chạy local trong container TEI
(profile docker "ai"). Mặc định: BAAI/bge-m3 (1024 chiều, đa ngôn ngữ) và
BAAI/bge-reranker-v2-m3.
"""

from dataclasses import dataclass

import httpx


@dataclass
class RerankHit:
    index: int
    score: float


class TEIEmbedder:
    def __init__(self, base_url: str, timeout: float = 60.0, batch_size: int = 16):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = await self._client.post("/embed", json={"inputs": batch, "normalize": True})
            resp.raise_for_status()
            vectors.extend(resp.json())
        return vectors

    async def health(self) -> bool:
        try:
            return (await self._client.get("/health", timeout=3.0)).status_code == 200
        except httpx.HTTPError:
            return False


class TEIReranker:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def rerank(self, query: str, texts: list[str]) -> list[RerankHit]:
        if not texts:
            return []
        resp = await self._client.post("/rerank", json={"query": query, "texts": texts})
        resp.raise_for_status()
        hits = [RerankHit(index=h["index"], score=h["score"]) for h in resp.json()]
        return sorted(hits, key=lambda h: h.score, reverse=True)
