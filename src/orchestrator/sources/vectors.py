"""Source 5: Qdrant vector search over arXiv abstracts on energy topics."""

from __future__ import annotations

import json
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient, models


class VectorStore:
    def __init__(self, client: AsyncQdrantClient, collection: str, embedding_model: str) -> None:
        self.client = client
        self.collection = collection
        self.embedder = TextEmbedding(embedding_model)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.embedder.embed(texts)]

    async def index_abstracts(self, path: Path) -> int:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        vectors = self._embed([f"{r['title']}. {r['summary']}" for r in rows])
        if await self.client.collection_exists(self.collection):
            await self.client.delete_collection(self.collection)
        await self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
        await self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(id=i, vector=v, payload=r)
                for i, (v, r) in enumerate(zip(vectors, rows, strict=True))
            ],
        )
        return len(rows)

    async def search(self, query: str, k: int = 5) -> list[dict]:
        hits = await self.client.query_points(self.collection, query=self._embed([query])[0], limit=k)
        return [
            {
                "title": h.payload["title"],
                "url": h.payload["url"],
                "published": h.payload["published"],
                "score": round(h.score, 3),
                "abstract": h.payload["summary"],
            }
            for h in hits.points
        ]
