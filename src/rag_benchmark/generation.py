"""Compatibility exports for the shared local Hugging Face runtime."""

from .shared.api_llm import APIModelConfig, OpenAICompatibleLLM
from .shared.local_llm import HFModelConfig, LocalLLM, base_prompt, rrag_prompt

__all__ = [
    "APIModelConfig", "HFModelConfig", "LocalLLM", "OpenAICompatibleLLM",
    "base_prompt", "rrag_prompt",
]

