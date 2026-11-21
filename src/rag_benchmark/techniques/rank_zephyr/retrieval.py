from __future__ import annotations

import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from ...shared.device import synchronize_cuda
from ...shared.records import RetrievedEvidence
from .config import fingerprint


class DenseEmbedding:
    def __init__(self, config, device: str):
        from sentence_transformers import SentenceTransformer
        from transformers.utils.hub import cached_file

        path = Path(config.embedding_model)
        local = path.is_absolute()
        self.model = SentenceTransformer(
            config.embedding_model, device=device, trust_remote_code=False,
            revision=None if local else config.embedding_revision, local_files_only=local,
        )
        self.model.max_seq_length = min(self.model.max_seq_length, config.embedding_max_length)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.device = device
        if local:
            files = sorted(
                (str(file.relative_to(path)), file.stat().st_size, file.stat().st_mtime_ns)
                for file in path.rglob("*") if file.is_file()
            )
            self.revision = fingerprint(files)
        else:
            transformer = self.model[0].auto_model
            self.revision = transformer.config._commit_hash
            if not self.revision:
                cached = Path(cached_file(
                    config.embedding_model, "config.json",
                    revision=config.embedding_revision, local_files_only=True,
                ))
                self.revision = cached.parent.name

    def encode(self, texts: list[str]):
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        synchronize_cuda(self.device, "RankZephyr dense embedding")
        return vectors


class DenseStore:
    def __init__(self, root: Path, namespace: str):
        self.namespace = namespace
        url = os.getenv("QDRANT_URL", "").strip()
        self.client = (
            QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None)
            if url else QdrantClient(path=str(root / "qdrant"))
        )

    def collection(self, language: str) -> str:
        return f"rank_zephyr_{self.namespace}_{language}"

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
                id=str(uuid5(NAMESPACE_URL, chunk.id)), vector=vector.tolist(),
                payload={
                    "document_id": chunk.document_id, "chunk_id": chunk.id, "text": chunk.text,
                    "metadata": {**chunk.metadata, "source_path": chunk.source_path, "title": chunk.title},
                },
            ) for chunk, vector in zip(chunks, vectors)
        ])

    def search(self, language: str, vector, limit: int) -> list[RetrievedEvidence]:
        hits = self.client.query_points(
            self.collection(language), query=vector.tolist(), limit=limit, with_payload=True,
            search_params=models.SearchParams(exact=True),
        ).points
        return [RetrievedEvidence(
            source_id=hit.payload["document_id"], chunk_id=hit.payload["chunk_id"],
            text=hit.payload["text"], rerank_score=float(hit.score), metadata=hit.payload["metadata"],
        ) for hit in hits]

    def close(self):
        self.client.close()
