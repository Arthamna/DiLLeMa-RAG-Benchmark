import pytest
from types import SimpleNamespace

from rag_benchmark.generation import (
    APIModelConfig, HFModelConfig, OpenAICompatibleLLM, base_prompt,
)


def test_hf_model_configuration_is_explicit_and_deterministic(monkeypatch):
    monkeypatch.delenv("HF_MODEL_NAME", raising=False)
    with pytest.raises(ValueError, match="HF_MODEL_NAME"):
        HFModelConfig.from_env()
    monkeypatch.setenv("HF_MODEL_NAME", "local/test-model")
    monkeypatch.setenv("HF_QUANTIZATION", "none")
    config = HFModelConfig.from_env()
    assert config.model_name == "local/test-model"
    assert config.max_new_tokens == 768
    assert config.fingerprint() == HFModelConfig.from_env().fingerprint()
    prompt = base_prompt("question", "evidence")
    assert "references" not in prompt.casefold()
    assert "temperature" not in prompt.casefold()


def test_openai_compatible_generator_uses_env_without_leaking_key(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_MODEL", "openrouter/free")
    monkeypatch.setenv("LLM_MODEL_SOURCE", "openrouter/free")
    monkeypatch.setenv("LLM_MAX_TOKENS", "321")
    config = APIModelConfig.from_env()
    assert APIModelConfig.is_configured()
    assert config.public_dict()["model"] == "openrouter/free"
    assert "api_key" not in config.public_dict()

    captured = {}

    class Completions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=" generated answer "),
            )])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    runtime = OpenAICompatibleLLM(config, client=client)
    assert runtime.generate("question", "evidence") == "generated answer"
    assert captured["model"] == "openrouter/free"
    assert captured["temperature"] == 0
    assert captured["max_tokens"] == 321
