import pytest

from rag_benchmark.generation import HFModelConfig, base_prompt


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
