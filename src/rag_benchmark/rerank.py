from __future__ import annotations

from typing import Sequence


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return [float(score) for score in self.model.predict([[question, text] for text in passages])]

