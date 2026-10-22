import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rag_benchmark.shared.config import GenerationSpec
from rag_benchmark.shared.device import CUDAStageError
from rag_benchmark.shared.local_llm import HFModelConfig, LocalLLM


@pytest.mark.parametrize("cuda", [True, False])
def test_unquantized_generator_loads_explicit_dtype_and_attention(monkeypatch, cuda):
    captured = {}

    def load_model(name, **kwargs):
        captured.update(kwargs)
        return "model"

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: "tokenizer"),
    ))
    assert LocalLLM._load(HFModelConfig(
        "snapshot", quantization="none", torch_dtype="bfloat16", attention_implementation="eager",
    )) == ("model", "tokenizer")
    assert captured["torch_dtype"] == ("bfloat16" if cuda else "float32")
    assert captured["attn_implementation"] == "eager"
    assert "quantization_config" not in captured


def test_generator_profile_reaches_runtime_and_fingerprint(monkeypatch):
    monkeypatch.setenv("HF_QUANTIZATION", "none")
    monkeypatch.setenv("HF_TORCH_DTYPE", "bfloat16")
    monkeypatch.setenv("HF_ATTENTION_IMPLEMENTATION", "eager")
    spec = GenerationSpec.from_dict({
        "backend": "huggingface", "model": "snapshot", "quantization": "4bit",
        "quantization_env": "HF_QUANTIZATION",
    })
    monkeypatch.setattr("rag_benchmark.shared.config.LocalLLM", lambda config: config)
    runtime_config = spec.runtime()
    assert runtime_config.quantization == "none"
    assert runtime_config.torch_dtype == "bfloat16"
    assert runtime_config.attention_implementation == "eager"
    assert spec.fingerprint() != replace(spec, attention_implementation="sdpa").fingerprint()
    assert spec.fingerprint() != replace(spec, torch_dtype="float16").fingerprint()
    # Explicit YAML settings still take precedence for dtype and attention.
    explicit = GenerationSpec.from_dict({
        "backend": "huggingface", "model": "snapshot",
        "torch_dtype": "float32", "attention_implementation": "sdpa",
    })
    assert explicit.torch_dtype == "float32"
    assert explicit.attention_implementation == "sdpa"


def test_generate_failure_reports_input_metadata_without_reusing_cuda(monkeypatch):
    failed = False

    def available():
        assert not failed, "must not query the failed CUDA context"
        return True

    class Tokens(dict):
        input_ids = SimpleNamespace(shape=(1, 4096))

        def to(self, device):
            assert not failed
            return self

    def generate(**kwargs):
        nonlocal failed
        failed = True
        raise RuntimeError("CUDA error: unspecified launch failure")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=available, synchronize=lambda device: None,
    )))
    parameter = SimpleNamespace(device="cuda:0", requires_grad=True)
    model = SimpleNamespace(
        parameters=lambda: iter([parameter]), eval=lambda: None, generate=generate,
    )
    tokenizer = SimpleNamespace(pad_token_id=0)
    runtime = LocalLLM(HFModelConfig(
        "snapshot", quantization="none", torch_dtype="bfloat16", attention_implementation="eager",
    ), model, tokenizer)
    monkeypatch.setattr(runtime, "tokenize", lambda prompt: Tokens())
    with pytest.raises(CUDAStageError, match="huggingface-generation") as caught:
        runtime.generate_prompt("question and evidence")
    assert "input_tokens=4096" in str(caught.value)
    assert "attention=eager" in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)
