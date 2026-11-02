from __future__ import annotations

from ...shared.device import synchronize_cuda
from .generator import model_options

_TRANSLATION = str.maketrans({
    **{chr(code): "" for code in (*range(0x01, 0x09), 0x0E, 0x0F, *range(0x11, 0x1C), 0xAD)},
    **{character: " " for character in "\u000b\u000c\u0085"},
    **{character: "-" for character in "‐‑⁃‒–—―−－⁻"},
    **{character: '"' for character in "“”„‟"},
    **{character: "'" for character in "‘’‚‛՚Ꞌꞌ＇`´′‵"},
    **{character: "/" for character in "⁄∕"},
    "″": "''", "‶": "''", "‴": "'''", "‷": "'''", "⁗": "''''", "…": "...",
})


def normalize_text(text: str) -> str:
    return text.translate(_TRANSLATION).replace(" . . . ", " ... ")


def prepare_text(text: str, config) -> str:
    if config.lowercase:
        text = text.lower()
    if config.normalize_text:
        text = normalize_text(text)
    return text


def passage_text(title: str, text: str) -> str:
    return f"{title} {text}" if title else text


class MContriever:
    def __init__(self, config, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        self.device = device
        options = model_options(config.retriever_model, config.retriever_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(config.retriever_model, **options)
        self.model = AutoModel.from_pretrained(
            config.retriever_model, add_pooling_layer=False, torch_dtype=torch.float32, **options,
        ).to(device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or config.retriever_revision
        self.vector_size = int(self.model.config.hidden_size)

    def _encode(self, texts: list[str], max_length: int):
        import numpy as np
        import torch

        batches = []
        for offset in range(0, len(texts), self.config.embedding_batch_size):
            batch = [prepare_text(text, self.config)
                     for text in texts[offset:offset + self.config.embedding_batch_size]]
            tokens = self.tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            synchronize_cuda(self.device, "mcontriever-embedding")
            batches.append(vectors.float().cpu().numpy())
        return np.concatenate(batches) if batches else np.empty((0, self.vector_size), dtype=np.float32)

    def encode_passages(self, texts: list[str]):
        return self._encode(texts, self.config.passage_max_length)

    def encode_query(self, text: str):
        return self._encode([text], self.config.query_max_length)[0]
