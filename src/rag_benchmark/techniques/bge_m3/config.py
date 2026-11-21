from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ...shared.config import ChunkingConfig

VARIANTS = (
    "bge_m3_dense",
    "bge_m3_sparse",
    "bge_m3_multi_vector",
    "bge_m3_dense_sparse",
    "bge_m3_all",
)


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class BgeM3Config:
    model: str = "BAAI/bge-m3"
    revision: str = "main"
    query_max_length: int = 512
    passage_max_length: int = 2048
    embedding_batch_size: int = 4
    use_fp16: bool = True
    trust_remote_code: bool = False
    dense_weight: float = 0.4
    sparse_weight: float = 0.2
    multi_vector_weight: float = 0.4

    @classmethod
    def from_dict(cls, value: dict, base: Path) -> "BgeM3Config | None":
        if not value or not value.get("enabled", False):
            return None
        values = {key: item for key, item in value.items() if key != "enabled"}
        model = str(values.get("model") or "BAAI/bge-m3").strip()
        path = Path(model).expanduser()
        if path.is_absolute() or model.startswith(("./", "../", "~")):
            path = path if path.is_absolute() else base / path
            if not (path / "config.json").is_file():
                raise FileNotFoundError(f"BGE-M3 snapshot config not found: {path / 'config.json'}")
            model = str(path.resolve())
        values["model"] = model
        config = cls(**values)
        if min(config.query_max_length, config.passage_max_length, config.embedding_batch_size) <= 0:
            raise ValueError("BGE-M3 lengths and batch size must be positive")
        if config.query_max_length > config.passage_max_length:
            raise ValueError("bge_m3.query_max_length cannot exceed passage_max_length")
        if not str(config.revision).strip():
            raise ValueError("bge_m3.revision is required")
        weights = config.weights
        if min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("BGE-M3 fusion weights must be non-negative with a positive sum")
        if config.dense_weight + config.sparse_weight <= 0:
            raise ValueError("BGE-M3 dense+sparse weights must have a positive sum")
        return config

    @property
    def weights(self) -> tuple[float, float, float]:
        return self.dense_weight, self.sparse_weight, self.multi_vector_weight

    def public_dict(self) -> dict:
        return asdict(self)

    def index_fingerprint(self, corpus_hash: str, chunking: ChunkingConfig) -> str:
        return fingerprint({
            "corpus": corpus_hash,
            "chunking": asdict(chunking),
            "model": self.model,
            "revision": self.revision,
            "passage_max_length": self.passage_max_length,
            "use_fp16": self.use_fp16,
            "outputs": ["dense", "learned_sparse", "colbert"],
        })

    def pipeline_fingerprint(self, corpus_hash: str, chunking: ChunkingConfig,
                             candidates: int, top_k: int) -> str:
        return fingerprint({
            "index": self.index_fingerprint(corpus_hash, chunking),
            "query_max_length": self.query_max_length,
            "weights": self.weights,
            "candidates_per_mode": candidates,
            "top_k": top_k,
            "candidate_modes": ["dense", "learned_sparse"],
            "prompt_version": 1,
        })
