from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ...shared.config import ChunkingConfig

VARIANTS = ("qwen3_embedding", "qwen3_embedding_rerank")


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    embedding_model: str
    reranker_model: str
    instruction: str
    embedding_dimension: int = 1024
    embedding_max_length: int = 2048
    reranker_max_length: int = 2048
    embedding_batch_size: int = 8
    reranker_batch_size: int = 1
    torch_dtype: str = "bfloat16"
    reranker_quantization: str = "none"
    attention_implementation: str = "sdpa"
    embedding_revision: str = "main"
    reranker_revision: str = "main"

    @classmethod
    def from_dict(cls, value: dict, base: Path) -> "Qwen3Config | None":
        if not value or not value.get("enabled", False):
            return None
        values = {key: item for key, item in value.items() if key != "enabled"}
        for key in ("embedding_model", "reranker_model"):
            model = str(values.get(key) or "").strip()
            if not model:
                raise ValueError(f"qwen3.{key} is required")
            path = Path(model).expanduser()
            if path.is_absolute() or model.startswith(("./", "../", "~")):
                path = path if path.is_absolute() else base / path
                if not (path / "config.json").is_file():
                    raise FileNotFoundError(f"Qwen3 snapshot config not found: {path / 'config.json'}")
                model = str(path.resolve())
            values[key] = model
        if not str(values.get("instruction") or "").strip():
            raise ValueError("qwen3.instruction is required")
        config = cls(**values)
        if not 32 <= config.embedding_dimension <= 4096:
            raise ValueError("qwen3.embedding_dimension must be between 32 and 4096")
        if min(config.embedding_max_length, config.reranker_max_length,
               config.embedding_batch_size, config.reranker_batch_size) <= 0:
            raise ValueError("Qwen3 lengths and batch sizes must be positive")
        if config.torch_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("Unsupported qwen3.torch_dtype")
        if config.reranker_quantization not in {"none", "4bit", "8bit"}:
            raise ValueError("qwen3.reranker_quantization must be none, 4bit or 8bit")
        if config.attention_implementation not in {"sdpa", "eager", "flash_attention_2"}:
            raise ValueError("Unsupported qwen3.attention_implementation")
        return config

    def public_dict(self) -> dict:
        return asdict(self)

    def index_fingerprint(self, corpus_hash: str, chunking: ChunkingConfig) -> str:
        return fingerprint({
            "corpus": corpus_hash, "chunking": asdict(chunking),
            "model": self.embedding_model, "revision": self.embedding_revision,
            "dimension": self.embedding_dimension, "max_length": self.embedding_max_length,
            "dtype": self.torch_dtype, "attention": self.attention_implementation,
            "pooling": "last-token", "normalize": True,
        })
