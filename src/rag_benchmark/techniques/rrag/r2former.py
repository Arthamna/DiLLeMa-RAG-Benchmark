from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:  # CLI/indexing remain usable without the optional RRAG extra.
    torch = None
    nn = None

from .config import RRAGConfig


def _require_torch() -> None:
    if torch is None:
        raise ImportError("R²AG requires the 'rrag' dependency extra (torch and transformers)")


if nn is not None:
    class R2Former(nn.Module):
        def __init__(self, config: RRAGConfig):
            super().__init__()
            self.config = config
            self.input_layer = nn.Linear(config.input_dim, config.d_model)
            self.position_embeddings = nn.Embedding(config.top_k, config.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_head,
                dropout=config.dropout,
                batch_first=True,
            )
            self.attention_layer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
            self.classifier_layer = nn.Linear(config.d_model, 1)

        def forward(self, features, labels=None, mask=None):
            if features.dim() == 2:
                features = features.unsqueeze(0)
            sequence_length = features.shape[1]
            if sequence_length > self.config.top_k:
                raise ValueError(f"R²-Former received {sequence_length} rows; configured maximum is {self.config.top_k}")
            positions = torch.arange(sequence_length, device=features.device).unsqueeze(0)
            hidden = self.input_layer(features) + self.position_embeddings(positions)
            padding_mask = ~mask.bool() if mask is not None else None
            hidden = self.attention_layer(hidden, src_key_padding_mask=padding_mask)
            logits = self.classifier_layer(hidden).squeeze(-1)
            loss = None
            if labels is not None:
                element_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.to(logits.dtype), reduction="none",
                )
                loss = (element_loss * mask).sum() / mask.sum() if mask is not None else element_loss.mean()
            return hidden, logits, loss


    class R2AGAdapter(nn.Module):
        def __init__(self, config: RRAGConfig, llm_hidden_size: int):
            super().__init__()
            self.config = config
            self.llm_hidden_size = llm_hidden_size
            self.r2former = R2Former(config)
            self.projection = nn.Linear(config.d_model, llm_hidden_size)

        def forward(self, features, labels=None, mask=None):
            hidden, logits, classification_loss = self.r2former(features, labels, mask)
            return self.projection(hidden), logits, classification_loss

        def save_checkpoint(self, directory, model_fingerprint: str, language: str) -> Path:
            target = Path(directory)
            target.mkdir(parents=True, exist_ok=True)
            path = target / "checkpoint.pt"
            torch.save({
                "state_dict": self.state_dict(),
                "rrag_config": asdict(self.config),
                "llm_hidden_size": self.llm_hidden_size,
                "model_fingerprint": model_fingerprint,
                "language": language,
            }, path)
            return path

        @classmethod
        def load_checkpoint(cls, path, expected_model_fingerprint: str | None = None, map_location="cpu"):
            payload = torch.load(Path(path), map_location=map_location, weights_only=True)
            fingerprint = str(payload["model_fingerprint"])
            if expected_model_fingerprint and fingerprint != expected_model_fingerprint:
                raise ValueError("RRAG checkpoint base-model fingerprint does not match HF_MODEL_NAME/config")
            model = cls(RRAGConfig(**payload["rrag_config"]), int(payload["llm_hidden_size"]))
            model.load_state_dict(payload["state_dict"])
            return model, payload
else:
    class R2Former:
        def __init__(self, *args, **kwargs):
            _require_torch()


    class R2AGAdapter:
        def __init__(self, *args, **kwargs):
            _require_torch()

