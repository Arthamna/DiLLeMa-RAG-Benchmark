from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from ...shared.records import RetrievedEvidence


class DenseStore:
    def __init__(self, root: Path, namespace: str, client=None):
        self.namespace = namespace
        self._owns_client = client is None
        if client is not None:
            self.client = client
        elif os.getenv("QDRANT_URL", "").strip():
            self.client = QdrantClient(
                url=os.environ["QDRANT_URL"], api_key=os.getenv("QDRANT_API_KEY") or None,
            )
        else:
            root.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(root / "qdrant"))

    def collection(self, language: str) -> str:
        return f"qwen3_{self.namespace}_{language}"

    def recreate(self, language: str, dimension: int):
        name = self.collection(language)
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            name, vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )

    def add(self, language: str, chunks, vectors):
        self.client.upsert(self.collection(language), points=[
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.id)), vector=vector.tolist(), payload=asdict(chunk),
            ) for chunk, vector in zip(chunks, vectors)
        ], wait=True)

    def search(self, language: str, vector, limit: int) -> list[RetrievedEvidence]:
        hits = self.client.query_points(
            self.collection(language), query=vector.tolist(), limit=limit, with_payload=True,
            search_params=models.SearchParams(exact=True),
        ).points
        return [RetrievedEvidence(
            source_id=hit.payload["document_id"], chunk_id=hit.payload["id"],
            text=hit.payload["text"], rerank_score=float(hit.score),
            metadata={**hit.payload["metadata"], "source_path": hit.payload["source_path"],
                      "title": hit.payload["title"], "dense_score": float(hit.score)},
        ) for hit in hits]

    def close(self):
        if self._owns_client:
            self.client.close()
