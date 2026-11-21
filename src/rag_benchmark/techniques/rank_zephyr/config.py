from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ...shared.config import ChunkingConfig


VARIANTS = ("rank_zephyr_dense", "rank_zephyr_rerank")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class RankZephyrConfig:
    embedding_model: str
    reranker_model: str
    reranker_revision: str
    worker_python: str
    embedding_revision: str = "main"
    embedding_batch_size: int = 8
    embedding_max_length: int = 512
    reranker_snapshot: str = ""

    @classmethod
    def from_dict(cls, value: dict, base: Path) -> RankZephyrConfig | None:
        if not value or not value.get("enabled", False):
            return None
        values = {key: item for key, item in value.items() if key != "enabled"}
        for key in ("embedding_model", "reranker_model", "reranker_revision"):
            if not str(values.get(key) or "").strip():
                raise ValueError(f"rank_zephyr.{key} is required")
        if not re.fullmatch(r"[0-9a-f]{7,40}", str(values["reranker_revision"])):
            raise ValueError("rank_zephyr.reranker_revision must be a Hugging Face commit hash")
        snapshot_env = values.pop("reranker_snapshot_env", "")
        snapshot = values.get("reranker_snapshot") or os.getenv(snapshot_env, "")
        if snapshot:
            values["reranker_snapshot"] = str((base / Path(snapshot).expanduser()).resolve())
        embedding = values["embedding_model"]
        if Path(embedding).is_absolute() or embedding.startswith(("./", "../", "~")):
            values["embedding_model"] = str((base / Path(embedding).expanduser()).resolve())
        # Virtualenv Python can be a symlink to the base interpreter on Ubuntu.
        python = values.get("worker_python") or "../../.venv-rank-llm/bin/python"
        values["worker_python"] = os.path.abspath(base / Path(python).expanduser())
        config = cls(**values)
        if min(config.embedding_batch_size, config.embedding_max_length) <= 0:
            raise ValueError("RankZephyr embedding length and batch size must be positive")
        return config

    def public_dict(self) -> dict:
        return asdict(self)

    def index_fingerprint(self, corpus_hash: str, resolved_revision: str) -> str:
        return fingerprint({
            "corpus": corpus_hash, "chunking": asdict(ChunkingConfig()),
            "model": self.embedding_model, "revision": resolved_revision,
            "max_length": self.embedding_max_length, "embedding": "sentence-transformers-normalized-v1",
        })
