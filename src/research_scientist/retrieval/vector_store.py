"""Qdrant: lưu vector của chunk + payload provenance; luôn lọc theo workspace (ACL)."""

from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

DENSE = "dense"


@dataclass
class VectorHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class ChunkVectorStore:
    def __init__(self, url: str, collection: str, dim: int):
        self._client = AsyncQdrantClient(url=url)
        self.collection = collection
        self.dim = dim

    async def ensure_collection(self) -> None:
        if await self._client.collection_exists(self.collection):
            return
        await self._client.create_collection(
            collection_name=self.collection,
            vectors_config={DENSE: models.VectorParams(size=self.dim, distance=models.Distance.COSINE)},
            # Sparse vector (BM25/SPLADE) sẽ thêm ở tuần 4 cho hybrid search
        )
        for field, schema in (
            ("workspace_id", models.PayloadSchemaType.KEYWORD),
            ("paper_id", models.PayloadSchemaType.KEYWORD),
            ("project_ids", models.PayloadSchemaType.KEYWORD),
            ("year", models.PayloadSchemaType.INTEGER),
        ):
            await self._client.create_payload_index(self.collection, field, schema)

    async def upsert(
        self, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]
    ) -> None:
        points = [
            models.PointStruct(id=i, vector={DENSE: v}, payload=p)
            for i, v, p in zip(ids, vectors, payloads, strict=True)
        ]
        await self._client.upsert(self.collection, points=points)

    async def search(
        self,
        vector: list[float],
        workspace_id: str,
        limit: int = 30,
        project_id: str | None = None,
        year_from: int | None = None,
    ) -> list[VectorHit]:
        must: list[models.Condition] = [
            models.FieldCondition(key="workspace_id", match=models.MatchValue(value=workspace_id))
        ]
        if project_id:
            must.append(
                models.FieldCondition(key="project_ids", match=models.MatchValue(value=project_id))
            )
        if year_from:
            must.append(models.FieldCondition(key="year", range=models.Range(gte=year_from)))
        res = await self._client.query_points(
            self.collection,
            query=vector,
            using=DENSE,
            query_filter=models.Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return [VectorHit(str(p.id), p.score, p.payload or {}) for p in res.points]

    async def health(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False
