from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RRAGConfig:
    top_k: int = 10
    input_dim: int = 3
    d_model: int = 256
    n_head: int = 4
    num_layers: int = 1
    dropout: float = 0.1
    epochs: int = 2
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 2

    @classmethod
    def from_env(cls) -> "RRAGConfig":
        return cls(
            top_k=int(os.getenv("RRAG_TOP_K", "10")),
            epochs=int(os.getenv("RRAG_EPOCHS", "2")),
            learning_rate=float(os.getenv("RRAG_LEARNING_RATE", "2e-4")),
            gradient_accumulation_steps=int(os.getenv("RRAG_GRADIENT_ACCUMULATION", "2")),
        )

