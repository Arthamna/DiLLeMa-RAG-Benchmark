from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class APIModelConfig:
    base_url: str
    api_key: str
    model: str
    model_source: str
    max_tokens: int = 768

    @classmethod
    def is_configured(cls) -> bool:
        return bool(os.getenv("LLM_API_KEY", "").strip())

    @classmethod
    def from_env(cls) -> "APIModelConfig":
        values = {
            "base_url": os.getenv("LLM_BASE_URL", "").strip(),
            "api_key": os.getenv("LLM_API_KEY", "").strip(),
            "model": os.getenv("LLM_MODEL", "").strip(),
        }
        missing = [f"LLM_{name.upper()}" for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"Missing required API generator variables: {', '.join(missing)}")
        return cls(
            **values,
            model_source=os.getenv("LLM_MODEL_SOURCE", values["model"]).strip() or values["model"],
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "768")),
        )

    def public_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key != "api_key"}

    def fingerprint(self) -> str:
        payload = json.dumps(self.public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OpenAICompatibleLLM:
    """Backend-neutral deterministic chat-completion runtime."""

    kind = "openai-compatible-api"

    def __init__(self, config: APIModelConfig, client=None):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Install the 'api' extra to use LLM_BASE_URL/LLM_API_KEY") from exc
            client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.config = config
        self.client = client

    @property
    def fingerprint(self) -> str:
        return self.config.fingerprint()

    def generate_prompt(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=self.config.max_tokens,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
