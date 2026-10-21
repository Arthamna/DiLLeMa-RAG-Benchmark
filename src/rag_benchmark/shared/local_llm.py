from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass

from .device import cuda_available

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HFModelConfig:
    model_name: str
    quantization: str = "4bit"
    trust_remote_code: bool = False
    max_input_tokens: int = 4096
    max_new_tokens: int = 768

    @classmethod
    def from_env(cls) -> "HFModelConfig":
        model_name = os.getenv("HF_MODEL_NAME", "").strip()
        if not model_name:
            raise ValueError("Missing required environment variable: HF_MODEL_NAME")
        quantization = os.getenv("HF_QUANTIZATION", "4bit").strip().casefold()
        if quantization not in {"4bit", "8bit", "none"}:
            raise ValueError("HF_QUANTIZATION must be one of: 4bit, 8bit, none")
        trust = os.getenv("HF_TRUST_REMOTE_CODE", "false").strip().casefold() in {"1", "true", "yes"}
        return cls(
            model_name=model_name,
            quantization=quantization,
            trust_remote_code=trust,
            max_input_tokens=int(os.getenv("HF_MAX_INPUT_TOKENS", "4096")),
            max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "768")),
        )

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LocalLLM:
    """Backend-neutral local model loader, tokenizer, and decoder."""

    kind = "huggingface-local"

    def __init__(self, config: HFModelConfig, model=None, tokenizer=None):
        self.config = config
        if model is None or tokenizer is None:
            model, tokenizer = self._load(config)
        self.model = model
        self.tokenizer = tokenizer
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    @staticmethod
    def _load(config: HFModelConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs = {"device_map": "auto", "trust_remote_code": config.trust_remote_code}
        if config.quantization != "none" and not cuda_available():
            LOGGER.warning(
                "CUDA GPU tidak terdeteksi; quantization %s dinonaktifkan dan LLM dimuat di CPU.",
                config.quantization,
            )
        elif config.quantization != "none":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError("Install the 'quantized' extra to use 4-bit or 8-bit loading") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=config.quantization == "4bit",
                load_in_8bit=config.quantization == "8bit",
            )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=config.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
        return model, tokenizer

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def fingerprint(self) -> str:
        return self.config.fingerprint()

    def tokenize(self, text: str):
        return self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.config.max_input_tokens, add_special_tokens=True,
        )

    def _decode_new_tokens(self, outputs, input_length: int) -> str:
        start = input_length if outputs.shape[1] > input_length else 0
        generated = outputs[0, start:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_prompt(self, prompt: str) -> str:
        tokens = self.tokenize(prompt).to(self.device)
        outputs = self.model.generate(
            **tokens, do_sample=False, max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self._decode_new_tokens(outputs, int(tokens.input_ids.shape[1]))

