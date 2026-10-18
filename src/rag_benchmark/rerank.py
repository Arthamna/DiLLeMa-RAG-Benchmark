from __future__ import annotations

import os
from typing import Sequence

from .shared.device import resolve_torch_device


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = resolve_torch_device("BGE reranker", device)
        self.batch_size = int(os.getenv("RERANK_BATCH_SIZE", "16"))
        self.model = CrossEncoder(model_name, device=self.device)

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return [float(score) for score in self.model.predict(
            [[question, text] for text in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )]

