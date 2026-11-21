from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from ...shared.config import ChunkingConfig

VARIANTS = ("self_rag_adaptive", "self_rag_always")
PROMPT_VERSION = 1


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _model_reference(value: str, base: Path, label: str) -> str:
    model = str(value or "").strip()
    if not model:
        raise ValueError(f"self_rag.{label} is required")
    path = Path(model).expanduser()
    if path.is_absolute() or model.startswith(("./", "../", "~")):
        path = path if path.is_absolute() else base / path
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Self-RAG {label} snapshot config not found: {path / 'config.json'}")
        return str(path.resolve())
    return model


@dataclass(frozen=True, slots=True)
class SelfRagConfig:
    model: str = "selfrag/selfrag_llama2_7b"
    revision: str = "main"
    quantization: str = "4bit"
    torch_dtype: str = "float16"
    max_input_tokens: int = 4096
    max_new_tokens: int = 100
    generation_batch_size: int = 1
    instruction: str = "Answer the question in the same language as the question."
    retriever_model: str = "facebook/mcontriever-msmarco"
    retriever_revision: str = "main"
    query_max_length: int = 512
    passage_max_length: int = 512
    embedding_batch_size: int = 32
    lowercase: bool = True
    normalize_text: bool = True
    ndocs: int = 5
    threshold: float = 0.2
    w_rel: float = 1.0
    w_sup: float = 1.0
    w_use: float = 0.5
    use_groundness: bool = True
    use_utility: bool = True
    use_seqscore: bool = True

    @classmethod
    def from_dict(cls, value: dict, base: Path) -> "SelfRagConfig | None":
        if not value or not value.get("enabled", False):
            return None
        values = {key: item for key, item in value.items() if key != "enabled"}
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown self_rag options: {', '.join(sorted(unknown))}")
        config = cls(**values)
        config = replace(
            config,
            model=_model_reference(config.model, base, "model"),
            retriever_model=_model_reference(config.retriever_model, base, "retriever_model"),
            quantization=str(config.quantization).strip().casefold(),
            instruction=str(config.instruction or "").strip(),
        )
        return config.validate()

    def validate(self) -> "SelfRagConfig":
        if self.quantization not in {"4bit", "8bit", "none"}:
            raise ValueError("self_rag.quantization must be 4bit, 8bit or none")
        if self.torch_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("self_rag.torch_dtype must be float16, bfloat16 or float32")
        positive = (
            self.max_input_tokens, self.max_new_tokens, self.generation_batch_size,
            self.query_max_length, self.passage_max_length, self.embedding_batch_size, self.ndocs,
        )
        if min(positive) <= 0:
            raise ValueError("Self-RAG lengths, batch sizes and ndocs must be positive")
        if not str(self.revision).strip() or not str(self.retriever_revision).strip():
            raise ValueError("self_rag.revision and self_rag.retriever_revision are required")
        if min(self.w_rel, self.w_sup, self.w_use) < 0:
            raise ValueError("Self-RAG reward weights must be non-negative")
        return self

    def public_dict(self) -> dict:
        return asdict(self)

    def generator_dict(self) -> dict:
        return {
            "model": self.model,
            "revision": self.revision,
            "quantization": self.quantization,
            "torch_dtype": self.torch_dtype,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
        }

    def model_fingerprint(self) -> str:
        return fingerprint(self.generator_dict())

    def decoding_dict(self) -> dict:
        return {"temperature": 0.0, "max_tokens": self.max_new_tokens, "do_sample": False}

    def index_fingerprint(self, corpus_hash: str, chunking: ChunkingConfig) -> str:
        return fingerprint({
            "corpus": corpus_hash,
            "chunking": asdict(chunking),
            "retriever_model": self.retriever_model,
            "retriever_revision": self.retriever_revision,
            "passage_max_length": self.passage_max_length,
            "lowercase": self.lowercase,
            "normalize_text": self.normalize_text,
            "pooling": "mean",
            "passage_format": "title text",
        })

    def pipeline_fingerprint(self, corpus_hash: str, chunking: ChunkingConfig, candidates: int) -> str:
        return fingerprint({
            "index": self.index_fingerprint(corpus_hash, chunking),
            "generator": self.generator_dict(),
            "query_max_length": self.query_max_length,
            "candidates": candidates,
            "ndocs": self.ndocs,
            "threshold": self.threshold,
            "weights": [self.w_rel, self.w_sup, self.w_use],
            "flags": [self.use_groundness, self.use_utility, self.use_seqscore],
            "instruction": self.instruction,
            "prompt_version": PROMPT_VERSION,
        })
