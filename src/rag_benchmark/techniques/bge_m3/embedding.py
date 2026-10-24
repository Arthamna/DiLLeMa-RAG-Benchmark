from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import models

from ...shared.device import synchronize_cuda


@dataclass(slots=True)
class BgeM3Batch:
    dense: list[np.ndarray]
    sparse: list[models.SparseVector]
    multi_vector: list[np.ndarray]

    def one(self) -> "BgeM3Encoding":
        if not (len(self.dense) == len(self.sparse) == len(self.multi_vector) == 1):
            raise ValueError("Expected exactly one BGE-M3 encoding")
        return BgeM3Encoding(self.dense[0], self.sparse[0], self.multi_vector[0])


@dataclass(slots=True)
class BgeM3Encoding:
    dense: np.ndarray
    sparse: models.SparseVector
    multi_vector: np.ndarray


def _sparse_vector(value) -> models.SparseVector:
    pairs = sorted((int(index), float(weight)) for index, weight in dict(value).items() if float(weight))
    return models.SparseVector(
        indices=[index for index, _ in pairs],
        values=[weight for _, weight in pairs],
    )


def _resolved_model(model: str, revision: str) -> tuple[str, str]:
    path = Path(model).expanduser()
    if path.is_dir():
        return str(path.resolve()), revision
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=model, revision=revision)).resolve()
    return str(snapshot), snapshot.name


class BgeM3Embedding:
    def __init__(self, config, device: str):
        from FlagEmbedding import BGEM3FlagModel

        self.config = config
        self.device = device
        model_path, self.revision = _resolved_model(config.model, config.revision)
        self.model = BGEM3FlagModel(
            model_path,
            use_fp16=config.use_fp16 and device.startswith("cuda"),
            devices=[device],
            trust_remote_code=config.trust_remote_code,
        )
        inner = getattr(self.model, "model", self.model)
        base = getattr(inner, "model", inner)
        model_config = getattr(base, "config", getattr(inner, "config", None))
        self.vector_size = int(getattr(model_config, "hidden_size", 1024))
        projection = getattr(inner, "colbert_linear", None)
        self.multi_vector_size = int(getattr(projection, "out_features", self.vector_size))

    def _encode(self, texts: list[str], max_length: int) -> BgeM3Batch:
        if not texts:
            return BgeM3Batch([], [], [])
        output = self.model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        synchronize_cuda(self.device, "bge-m3-embedding")
        dense = [np.asarray(item, dtype=np.float32) for item in output["dense_vecs"]]
        sparse = [_sparse_vector(item) for item in output["lexical_weights"]]
        multi = [np.asarray(item, dtype=np.float32) for item in output["colbert_vecs"]]
        if not (len(dense) == len(sparse) == len(multi) == len(texts)):
            raise RuntimeError("BGE-M3 returned an incomplete encoding batch")
        return BgeM3Batch(dense, sparse, multi)

    def encode_passages(self, texts: list[str]) -> BgeM3Batch:
        return self._encode(texts, self.config.passage_max_length)

    def encode_query(self, text: str) -> BgeM3Encoding:
        # BGE-M3 does not require a retrieval instruction prefix.
        return self._encode([text], self.config.query_max_length).one()
