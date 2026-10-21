from __future__ import annotations

from pathlib import Path

from ...shared.device import synchronize_cuda


def model_options(model: str, revision: str) -> dict:
    return {"local_files_only": True} if Path(model).is_dir() else {"revision": revision}


def last_token_pool(hidden, attention_mask):
    if bool((attention_mask[:, -1] == 1).all()):
        return hidden[:, -1]
    import torch

    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


class Qwen3Embedding:
    def __init__(self, config, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        self.device = device
        options = model_options(config.embedding_model, config.embedding_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.embedding_model, padding_side="left", trust_remote_code=False, **options,
        )
        self.model = AutoModel.from_pretrained(
            config.embedding_model, trust_remote_code=False,
            torch_dtype=getattr(torch, config.torch_dtype) if device.startswith("cuda") else torch.float32,
            attn_implementation=config.attention_implementation if device.startswith("cuda") else "sdpa", **options,
        ).to(device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or config.embedding_revision
        if config.embedding_dimension > self.model.config.hidden_size:
            raise ValueError("Qwen3 embedding dimension exceeds model hidden size")
        self.vector_size = config.embedding_dimension

    def _encode(self, texts):
        import numpy as np
        import torch
        import torch.nn.functional as F

        batches = []
        for offset in range(0, len(texts), self.config.embedding_batch_size):
            tokens = self.tokenizer(
                texts[offset:offset + self.config.embedding_batch_size], padding=True,
                truncation=True, max_length=self.config.embedding_max_length, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                output = self.model(**tokens, use_cache=False)
                vectors = last_token_pool(output.last_hidden_state, tokens["attention_mask"])
                vectors = F.normalize(vectors[:, :self.vector_size].float(), p=2, dim=1)
            synchronize_cuda(self.device, "qwen3-embedding")
            batches.append(vectors.cpu().numpy())
        return np.concatenate(batches) if batches else np.empty((0, self.vector_size), dtype=np.float32)

    def encode_passages(self, texts):
        return self._encode(texts)

    def encode_query(self, question):
        return self._encode([f"Instruct: {self.config.instruction}\nQuery: {question}"])[0]
