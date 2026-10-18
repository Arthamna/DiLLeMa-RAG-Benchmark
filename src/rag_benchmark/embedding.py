from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from .shared.device import resolve_torch_device


class E5Embedding:
    """Normalized multilingual E5 embeddings with required query/passage prefixes."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = resolve_torch_device("E5 embedding", device)
        self.batch_size = int(os.getenv("EMBED_BATCH_SIZE", "32"))
        self.model = SentenceTransformer(model_name, device=self.device)

    @property
    def vector_size(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            [f"passage: {text}" for text in texts],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            f"query: {text}",
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

