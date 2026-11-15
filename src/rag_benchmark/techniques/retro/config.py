from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _resolve(base: Path, value: Any, field: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{field} is required")
    candidate = Path(raw).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    retrieval_endpoint: str = "127.0.0.1:50051"
    inference_endpoint: str = "127.0.0.1:50052"
    retrieval_index_root: str = "/data/indexes"
    timeout_seconds: float = 120.0
    protocol_version: int = 1
    max_message_mb: int = 64

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ServiceConfig":
        defaults = cls()
        config = cls(
            retrieval_endpoint=str(value.get("retrieval_endpoint") or defaults.retrieval_endpoint),
            inference_endpoint=str(value.get("inference_endpoint") or defaults.inference_endpoint),
            retrieval_index_root=str(value.get("retrieval_index_root") or defaults.retrieval_index_root),
            timeout_seconds=float(value.get("timeout_seconds", defaults.timeout_seconds)),
            protocol_version=int(value.get("protocol_version", defaults.protocol_version)),
            max_message_mb=int(value.get("max_message_mb", defaults.max_message_mb)),
        )
        if not config.retrieval_endpoint or not config.inference_endpoint or not config.retrieval_index_root:
            raise ValueError("services endpoints and retrieval_index_root cannot be empty")
        if config.timeout_seconds <= 0 or config.max_message_mb < 4:
            raise ValueError("services timeout must be positive and max_message_mb at least 4")
        return config


@dataclass(frozen=True, slots=True)
class RetroConfig:
    checkpoint: Path
    model_config: Path
    onnx_encoder: Path
    onnx_decoder: Path
    tokenizer: str = "t5-base"
    retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 64
    num_neighbours: int = 2
    continuation_chunks: int = 1
    retrieval_interval: int = 64
    nprobe: int = 16
    max_new_tokens: int = 128

    @classmethod
    def from_dict(cls, value: dict[str, Any], base: Path) -> "RetroConfig":
        config = cls(
            checkpoint=_resolve(base, value.get("checkpoint"), "retro.checkpoint"),
            model_config=_resolve(base, value.get("model_config"), "retro.model_config"),
            onnx_encoder=_resolve(base, value.get("onnx_encoder"), "retro.onnx_encoder"),
            onnx_decoder=_resolve(base, value.get("onnx_decoder"), "retro.onnx_decoder"),
            tokenizer=str(value.get("tokenizer") or "t5-base"),
            retriever_model=str(value.get("retriever_model") or "sentence-transformers/all-MiniLM-L6-v2"),
            chunk_size=int(value.get("chunk_size", 64)),
            num_neighbours=int(value.get("num_neighbours", 2)),
            continuation_chunks=int(value.get("continuation_chunks", 1)),
            retrieval_interval=int(value.get("retrieval_interval", 64)),
            nprobe=int(value.get("nprobe", 16)),
            max_new_tokens=int(value.get("max_new_tokens", 128)),
        )
        if config.chunk_size != 64:
            raise ValueError("The upstream RETRO checkpoint requires retro.chunk_size=64")
        if min(config.num_neighbours, config.retrieval_interval, config.nprobe, config.max_new_tokens) <= 0:
            raise ValueError("RETRO numeric settings must be positive")
        if config.continuation_chunks < 0:
            raise ValueError("retro.continuation_chunks cannot be negative")
        return config

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("checkpoint", "model_config", "onnx_encoder", "onnx_decoder"):
            value[key] = str(value[key])
        return value

    def fingerprint(self) -> str:
        raw = json.dumps(self.public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class PaperEvaluationConfig:
    enabled: bool = True
    holdout_fraction: float = 0.10
    seed: int = 42
    max_windows_per_language: int = 32
    window_tokens: int = 1024
    latency_warmups: int = 1
    latency_repetitions: int = 5
    intervals: tuple[int, ...] = (8, 16, 32, 64)
    nprobes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PaperEvaluationConfig":
        defaults = cls()
        config = cls(
            enabled=bool(value.get("enabled", True)),
            holdout_fraction=float(value.get("holdout_fraction", defaults.holdout_fraction)),
            seed=int(value.get("seed", defaults.seed)),
            max_windows_per_language=int(value.get("max_windows_per_language", defaults.max_windows_per_language)),
            window_tokens=int(value.get("window_tokens", defaults.window_tokens)),
            latency_warmups=int(value.get("latency_warmups", defaults.latency_warmups)),
            latency_repetitions=int(value.get("latency_repetitions", defaults.latency_repetitions)),
            intervals=tuple(int(item) for item in value.get("intervals", defaults.intervals)),
            nprobes=tuple(int(item) for item in value.get("nprobes", defaults.nprobes)),
        )
        if not 0 < config.holdout_fraction < 1:
            raise ValueError("paper_evaluation.holdout_fraction must be between 0 and 1")
        if min(config.max_windows_per_language, config.window_tokens, config.latency_repetitions) <= 0:
            raise ValueError("paper evaluation sizes must be positive")
        if any(value <= 0 or 64 % value for value in config.intervals):
            raise ValueError("paper intervals must be positive divisors of 64")
        if not config.nprobes or any(value <= 0 for value in config.nprobes):
            raise ValueError("paper nprobes must contain positive integers")
        return config
