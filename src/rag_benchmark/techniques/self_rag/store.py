from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from ...shared.records import RetrievedEvidence


class SelfRagStore:
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
        return f"self_rag_{self.namespace}_{language}"

    def recreate(self, language: str, dimension: int):
        name = self.collection(language)
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            name, vectors_config=models.VectorParams(size=dimension, distance=models.Distance.DOT),
        )

    def add(self, language: str, chunks, vectors):
        if len(chunks) != len(vectors):
            raise ValueError("Self-RAG chunks and vectors have different lengths")
        self.client.upsert(self.collection(language), points=[
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{self.collection(language)}:{chunk.id}")),
                vector=vector.tolist(), payload=asdict(chunk),
            ) for chunk, vector in zip(chunks, vectors)
        ], wait=True)

    def search(self, language: str, vector, limit: int) -> list[RetrievedEvidence]:
        hits = self.client.query_points(
            self.collection(language), query=vector.tolist(), limit=limit, with_payload=True,
        ).points
        return [RetrievedEvidence(
            source_id=str(hit.payload["document_id"]), chunk_id=str(hit.payload["id"]),
            text=str(hit.payload["text"]), rerank_score=float(hit.score),
            metadata={**dict(hit.payload.get("metadata") or {}),
                      "source_path": hit.payload.get("source_path"),
                      "title": hit.payload.get("title"),
                      "aliases": hit.payload.get("aliases", []),
                      "retrieval_score": float(hit.score)},
        ) for hit in hits]

    def close(self):
        if self._owns_client:
            self.client.close()
