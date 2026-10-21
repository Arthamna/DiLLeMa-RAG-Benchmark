from __future__ import annotations

import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Fusion,
    FusionQuery,
    Modifier,
    PointStruct,
    Prefetch,
    SparseVectorParams,
    VectorParams,
)

from ...shared.records import ChunkRecord
from .sparse import encode_sparse

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


class QdrantStore:
    """Persistent local Qdrant by default, or remote Qdrant through environment."""

    def __init__(self, index_root: str | Path, client: QdrantClient | None = None):
        self.index_root = Path(index_root)
        self._owns_client = client is None
        if client is not None:
            self.client = client
            return
        url = os.getenv("QDRANT_URL", "").strip()
        api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.index_root.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.index_root / "qdrant"))

    def close(self) -> None:
        if self._owns_client and hasattr(self.client, "close"):
            self.client.close()

    def recreate_collection(self, name: str, vector_size: int) -> None:
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            collection_name=name,
            vectors_config={DENSE_VECTOR: VectorParams(size=vector_size, distance=Distance.COSINE)},
            sparse_vectors_config={
                SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF),
            },
        )

    def add_chunks(self, name: str, chunks: list[ChunkRecord], vectors, batch_size: int = 128) -> None:
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            dense_batch = vectors[offset : offset + batch_size]
            points: list[PointStruct] = []
            for chunk, dense in zip(batch, dense_batch):
                payload = {
                    "document": chunk.text,
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "dataset": chunk.dataset,
                    "language": chunk.language,
                    "evidence_text": chunk.evidence_text,
                    "parent_window": chunk.parent_window,
                    "parent_id": chunk.parent_id,
                    "source_path": chunk.source_path,
                    "title": chunk.title,
                    "metadata": chunk.metadata,
                }
                vector = dense.tolist() if hasattr(dense, "tolist") else list(dense)
                points.append(
                    PointStruct(
                        id=str(uuid5(NAMESPACE_URL, f"{name}:{chunk.id}")),
                        vector={DENSE_VECTOR: vector, SPARSE_VECTOR: encode_sparse(chunk.text)},
                        payload=payload,
                    )
                )
            self.client.upsert(collection_name=name, points=points, wait=True)

    def search(self, name: str, query_vector, query_text: str, limit: int) -> list:
        dense = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        response = self.client.query_points(
            collection_name=name,
            prefetch=[
                Prefetch(query=dense, using=DENSE_VECTOR, limit=limit),
                Prefetch(query=encode_sparse(query_text), using=SPARSE_VECTOR, limit=limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return list(response.points)

    def count(self, name: str) -> int:
        return int(self.client.count(collection_name=name, exact=True).count)

