"""Compatibility exports for the shared local Hugging Face runtime."""

from .shared.local_llm import HFModelConfig, LocalLLM, base_prompt, rrag_prompt

__all__ = ["HFModelConfig", "LocalLLM", "base_prompt", "rrag_prompt"]

