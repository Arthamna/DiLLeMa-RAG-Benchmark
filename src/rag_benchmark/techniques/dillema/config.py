from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

from ...shared.config import ChunkingConfig


@dataclass(frozen=True, slots=True)
class PipelineConfig(ChunkingConfig):
    retrieval_candidates: int = 40
    rerank_min_score: float = 0.05
    rerank_relative_floor: float = 0.5
    normal_top_results: int = 8
    max_parents: int = 4
    listing_top_results: int = 16
    listing_relative_floor: float = 0.3
    listing_max_parents: int = 8
    embedding_model: str = "intfloat/multilingual-e5-base"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    temperature: float = 0.0
    max_tokens: int = 768

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        defaults = cls()
        return cls(
            embedding_model=os.getenv("EMBED_MODEL_NAME", defaults.embedding_model),
            reranker_model=os.getenv("RERANK_MODEL_NAME", defaults.reranker_model),
        )

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
