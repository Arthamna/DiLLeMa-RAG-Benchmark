from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PipeRagConfig:
    retrieval_interval: int = 64
    stale_steps: int = 64
    dynamic_nprobe: bool = True
    search_latency_budget_discount: float = 1.0
    extra_overhead_ms: float = 10.0
    min_nprobe: int = 1
    max_nprobe: int = 64

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PipeRagConfig":
        defaults = cls()
        interval = int(value.get("retrieval_interval", defaults.retrieval_interval))
        config = cls(
            retrieval_interval=interval,
            stale_steps=int(value.get("stale_steps", interval)),
            dynamic_nprobe=bool(value.get("dynamic_nprobe", defaults.dynamic_nprobe)),
            search_latency_budget_discount=float(
                value.get("search_latency_budget_discount", defaults.search_latency_budget_discount)
            ),
            extra_overhead_ms=float(value.get("extra_overhead_ms", defaults.extra_overhead_ms)),
            min_nprobe=int(value.get("min_nprobe", defaults.min_nprobe)),
            max_nprobe=int(value.get("max_nprobe", defaults.max_nprobe)),
        )
        if config.retrieval_interval <= 0 or 64 % config.retrieval_interval:
            raise ValueError("piperag.retrieval_interval must be a positive divisor of 64")
        if config.stale_steps != config.retrieval_interval:
            raise ValueError("Upstream PipeRAG requires stale_steps equal to retrieval_interval")
        if not 0 < config.search_latency_budget_discount <= 1:
            raise ValueError("search_latency_budget_discount must be in (0, 1]")
        if config.min_nprobe <= 0 or config.max_nprobe < config.min_nprobe:
            raise ValueError("PipeRAG nprobe bounds are invalid")
        return config

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)
