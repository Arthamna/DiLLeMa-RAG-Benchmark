from __future__ import annotations

from dataclasses import dataclass

from ..retro.indexing import clamp_nprobe


@dataclass(frozen=True, slots=True)
class PerformanceProfile:
    decoder_latency_ms: tuple[float, ...]
    encoder_latency_ms: float
    embedding_latency_ms: float
    search_latency_ms_per_nprobe: float
    extra_overhead_ms: float = 10.0

    def choose_nprobe(
        self, seq_len: int, interval: int, nlist: int, minimum: int = 1,
        maximum: int = 64, discount: float = 1.0,
    ) -> int:
        if not self.decoder_latency_ms:
            return clamp_nprobe(minimum, nlist)
        budget = 0.0
        for position in range(seq_len, seq_len + interval):
            budget += self.decoder_latency_ms[min(position, len(self.decoder_latency_ms) - 1)]
        available = max(0.0, budget - self.embedding_latency_ms - self.extra_overhead_ms) * discount
        predicted = int(available / max(self.search_latency_ms_per_nprobe, 1e-9))
        return clamp_nprobe(max(minimum, min(maximum, predicted)), nlist)
