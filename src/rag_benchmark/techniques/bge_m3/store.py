from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from ...shared.records import RetrievedEvidence
from .embedding import BgeM3Batch, BgeM3Encoding

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "learned_sparse"
MULTI_VECTOR = "colbert"


@dataclass(slots=True)
class ScoredCandidate:
    evidence: RetrievedEvidence
    dense_score: float
    sparse_score: float
    multi_vector_score: float


def _sparse_dot(left: models.SparseVector, right: models.SparseVector) -> float:
    left_values = dict(zip(left.indices, left.values))
    return float(sum(left_values.get(index, 0.0) * value
                     for index, value in zip(right.indices, right.values)))


def _multi_vector_score(query, passage) -> float:
    query_array = np.asarray(query, dtype=np.float32)
    passage_array = np.asarray(passage, dtype=np.float32)
    if query_array.size == 0 or passage_array.size == 0:
        return 0.0
    # BGE-M3 uses the mean of per-query-token MaxSim values.
    return float((query_array @ passage_array.T).max(axis=1).mean())


class BgeM3Store:
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
        return f"bge_m3_{self.namespace}_{language}"

    def recreate(self, language: str, dense_size: int, multi_vector_size: int):
        name = self.collection(language)
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=dense_size, distance=models.Distance.DOT),
                MULTI_VECTOR: models.VectorParams(
                    size=multi_vector_size,
                    distance=models.Distance.DOT,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            },
            # These are learned lexical weights; applying Qdrant IDF would double-weight them.
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
        )

    def add(self, language: str, chunks, encoded: BgeM3Batch):
        if not (len(chunks) == len(encoded.dense) == len(encoded.sparse) == len(encoded.multi_vector)):
            raise ValueError("BGE-M3 chunks and encodings have different lengths")
        self.client.upsert(self.collection(language), points=[
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{self.collection(language)}:{chunk.id}")),
                vector={
                    DENSE_VECTOR: dense.tolist(),
                    SPARSE_VECTOR: sparse,
                    MULTI_VECTOR: multi.tolist(),
                },
                payload=asdict(chunk),
            )
            for chunk, dense, sparse, multi in zip(
                chunks, encoded.dense, encoded.sparse, encoded.multi_vector,
            )
        ], wait=True)

    def _search(self, language: str, query, using: str, limit: int):
        return self.client.query_points(
            self.collection(language), query=query, using=using, limit=limit,
            with_payload=True, search_params=models.SearchParams(exact=True),
        ).points

    def search(self, language: str, query: BgeM3Encoding, limit: int) -> list[ScoredCandidate]:
        dense_hits = self._search(language, query.dense.tolist(), DENSE_VECTOR, limit)
        sparse_hits = self._search(language, query.sparse, SPARSE_VECTOR, limit)
        ordered_ids = list(dict.fromkeys([hit.id for hit in dense_hits] + [hit.id for hit in sparse_hits]))
        if not ordered_ids:
            return []
        points = self.client.retrieve(
            self.collection(language), ids=ordered_ids, with_payload=True, with_vectors=True,
        )
        by_id = {point.id: point for point in points}
        result = []
        for point_id in ordered_ids:
            point = by_id[point_id]
            payload = dict(point.payload or {})
            vectors = point.vector or {}
            dense_score = float(np.dot(query.dense, np.asarray(vectors[DENSE_VECTOR], dtype=np.float32)))
            sparse_score = _sparse_dot(query.sparse, vectors[SPARSE_VECTOR])
            multi_score = _multi_vector_score(query.multi_vector, vectors[MULTI_VECTOR])
            metadata = {
                **dict(payload.get("metadata") or {}),
                "source_path": payload.get("source_path"),
                "title": payload.get("title"),
                "aliases": payload.get("aliases", []),
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "multi_vector_score": multi_score,
            }
            evidence = RetrievedEvidence(
                source_id=str(payload.get("document_id") or ""),
                chunk_id=str(payload.get("id") or ""),
                text=str(payload.get("text") or ""),
                rerank_score=dense_score,
                metadata=metadata,
            )
            result.append(ScoredCandidate(evidence, dense_score, sparse_score, multi_score))
        return result

    def close(self):
        if self._owns_client:
            self.client.close()
