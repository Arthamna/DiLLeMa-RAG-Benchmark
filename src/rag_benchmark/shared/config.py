from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from .api_llm import APIModelConfig, OpenAICompatibleLLM
from .local_llm import HFModelConfig, LocalLLM


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120
    min_chunk_chars: int = 150
    parent_window_chars: int = 5000


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    backend: str
    model: str
    temperature: float
    max_tokens: int
    base_url: str = ""
    api_key_env: str = "LLM_API_KEY"
    quantization: str = "4bit"
    trust_remote_code: bool = False
    max_input_tokens: int = 4096
    torch_dtype: str | None = None
    attention_implementation: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationSpec":
        backend = str(value.get("backend") or "").strip().casefold()
        if backend not in {"huggingface", "openai-compatible"}:
            raise ValueError("generator.backend must be huggingface or openai-compatible")
        model = str(value.get("model") or "").strip()
        model_env = str(value.get("model_env") or "").strip()
        if not model and model_env:
            model = os.getenv(model_env, "").strip()
        if not model:
            raise ValueError("Generator model is empty; set generator.model or its model_env")
        base_url = str(value.get("base_url") or "").strip()
        base_url_env = str(value.get("base_url_env") or "LLM_BASE_URL").strip()
        if not base_url and base_url_env:
            base_url = os.getenv(base_url_env, "").strip()
        api_key_env = str(value.get("api_key_env") or "LLM_API_KEY").strip()
        temperature = float(value.get("temperature", 0.0))
        if temperature != 0.0:
            raise ValueError("Controlled benchmark requires generator.temperature=0")
        max_tokens_env = str(value.get("max_tokens_env") or "").strip()
        max_input_tokens_env = str(value.get("max_input_tokens_env") or "").strip()
        quantization_env = str(value.get("quantization_env") or "").strip()
        quantization = str(value.get("quantization") or os.getenv("HF_QUANTIZATION", "4bit"))
        if quantization_env:
            quantization = os.getenv(quantization_env, quantization)
        return cls(
            backend=backend,
            model=model,
            temperature=temperature,
            max_tokens=int(os.getenv(max_tokens_env, str(value.get("max_tokens", 768)))) if max_tokens_env else int(value.get("max_tokens", 768)),
            base_url=base_url,
            api_key_env=api_key_env,
            quantization=quantization,
            trust_remote_code=bool(value.get("trust_remote_code", False)),
            max_input_tokens=(
                int(os.getenv(max_input_tokens_env, str(value.get("max_input_tokens", 4096))))
                if max_input_tokens_env else int(value.get("max_input_tokens", 4096))
            ),
            torch_dtype=(value.get("torch_dtype") or os.getenv("HF_TORCH_DTYPE") or None)
            if backend == "huggingface" else None,
            attention_implementation=(
                value.get("attention_implementation") or os.getenv("HF_ATTENTION_IMPLEMENTATION") or None
            ) if backend == "huggingface" else None,
        )

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("torch_dtype", "attention_implementation"):
            if value[key] is None:
                del value[key]
        return value

    def fingerprint(self) -> str:
        payload = json.dumps(self.public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def decoding_dict(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens, "do_sample": False}

    def runtime(self):
        if self.backend == "openai-compatible":
            if not self.base_url:
                raise ValueError("OpenAI-compatible generation requires base_url or base_url_env")
            if not os.getenv(self.api_key_env, "").strip():
                raise ValueError(
                    f"OpenAI-compatible generation requires environment variable {self.api_key_env}"
                )
            return OpenAICompatibleLLM(APIModelConfig(
                base_url=self.base_url,
                api_key=os.environ[self.api_key_env],
                model=self.model,
                model_source=self.model,
                max_tokens=self.max_tokens,
            ))
        return LocalLLM(HFModelConfig(
            model_name=self.model,
            quantization=self.quantization,
            trust_remote_code=self.trust_remote_code,
            max_input_tokens=self.max_input_tokens,
            max_new_tokens=self.max_tokens,
            torch_dtype=self.torch_dtype,
            attention_implementation=self.attention_implementation,
        ))


def collection_name(dataset: str, language: str | None = None) -> str:
    if dataset == "its":
        return "its"
    if dataset == "uniqa" and language in {"en", "it"}:
        return f"uniqa_{language}"
    raise ValueError(f"Unsupported dataset/language: {dataset}/{language}")
