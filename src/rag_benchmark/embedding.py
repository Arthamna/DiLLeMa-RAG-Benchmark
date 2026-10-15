from __future__ import annotations

from typing import Sequence

import numpy as np


class E5Embedding:
    """Normalized multilingual E5 embeddings with required query/passage prefixes."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    @property
    def vector_size(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            [f"passage: {text}" for text in texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            f"query: {text}",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

