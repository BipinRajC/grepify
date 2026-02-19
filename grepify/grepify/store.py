"""Qdrant vector store operations."""

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .models import Chunk


class QdrantStore:
    def __init__(self, url: str = "http://localhost:6333", api_key: str | None = None):
        kwargs = {"url": url}
        if api_key:
            kwargs["api_key"] = api_key
        self.client = QdrantClient(**kwargs)

    def create_collection(self, name: str, dim: int = 768) -> None:
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"created collection '{name}' (dim={dim})")

    def upload(
        self,
        collection: str,
        chunks: list[Chunk],
        vectors: np.ndarray,
        batch_size: int = 100,
    ) -> None:
        points = [
            PointStruct(
                id=idx,
                vector=vec.tolist(),
                payload={
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "tier": chunk.tier,
                    "source_type": chunk.source_type,
                    **chunk.metadata,
                },
            )
            for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]

        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self.client.upsert(collection_name=collection, points=batch)
            print(f"  uploaded {start + len(batch)}/{len(points)}")

    def search(
        self,
        collection: str,
        query_vector: np.ndarray,
        limit: int = 10,
        tier: str | None = None,
    ) -> list[dict]:
        qfilter = None
        if tier:
            qfilter = Filter(
                must=[FieldCondition(key="tier", match=MatchValue(value=tier))]
            )

        response = self.client.query_points(
            collection_name=collection,
            query=query_vector.tolist(),
            limit=limit,
            query_filter=qfilter,
        )

        return [{"score": p.score, **p.payload} for p in response.points]

    def count(self, collection: str) -> int:
        return self.client.get_collection(collection).points_count
