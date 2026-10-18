import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from rag_benchmark.benchmark import (
    RANKIFY_PRESETS,
    ExperimentSpec,
    GenerationSpec,
    _convert_worker_rows,
    _question_payload,
)
from rag_benchmark.cli import build_parser, run_compare_techniques
from rag_benchmark.metrics import write_jsonl
from rag_benchmark.models import QARecord
from rag_benchmark.techniques import TECHNIQUES, get_technique

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rankify_worker" / "src"))
from rag_benchmark_rankify.worker import (  # noqa: E402
    PRESETS,
    _capabilities,
    _custom_prompt,
    _generation_kwargs,
    _generator,
)


def _fake_rankify(monkeypatch):
    package = ModuleType("rankify")
    package.__path__ = []
    generator_package = ModuleType("rankify.generator")
    generator_package.__path__ = []
    generator_module = ModuleType("rankify.generator.generator")
    models_package = ModuleType("rankify.models")
    models_package.__path__ = []
    reranking_module = ModuleType("rankify.models.reranking")
    retrievers_package = ModuleType("rankify.retrievers")
    retrievers_package.__path__ = []
    retriever_module = ModuleType("rankify.retrievers.retriever")
    prompt_module = ModuleType("rankify.generator.prompt_template")

    class Generator:
        last_kwargs = None

        def __init__(self, **kwargs):
            type(self).last_kwargs = kwargs

    class Reranking:
        pass

    class Retriever:
        @staticmethod
        def supported_methods():
            return ("bm25", "bge")

    generator_module.Generator = Generator
    reranking_module.Reranking = Reranking
    retriever_module.Retriever = Retriever
    prompt_module.DEFAULT_PROMPTS = {"in-context-ralm": "native RALM prompt"}
    prompt_module.PromptTemplate = SimpleNamespace(IN_CONTEXT_RALM="in-context-ralm")
    for name, module in {
        "rankify": package,
        "rankify.generator": generator_package,
        "rankify.generator.generator": generator_module,
        "rankify.generator.prompt_template": prompt_module,
        "rankify.models": models_package,
        "rankify.models.reranking": reranking_module,
        "rankify.retrievers": retrievers_package,
        "rankify.retrievers.retriever": retriever_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.1.4")
    return Generator


def test_rrag_is_absent_from_registry_and_cli():
    assert tuple(TECHNIQUES) == ("dillema",)
    with pytest.raises(ValueError, match="Unsupported"):
        get_technique("rrag")
    parser = build_parser()
    help_text = parser.format_help()
    assert "train" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--technique", "rrag"])


def test_generation_and_experiment_models_are_configuration_driven(tmp_path, monkeypatch):
    source = tmp_path / "dataset"
    source.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setenv("HF_MODEL_NAME", "small/model-a")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join([
            "dataset: its",
            "source: dataset",
            "index_root: indexes",
            "rankify_python: python",
            "generator:",
            "  backend: huggingface",
            "  model_env: HF_MODEL_NAME",
            "  temperature: 0",
            "  max_tokens: 64",
            "models:",
            "  dense_multilingual: configurable/dense",
            "  reranker_multilingual: configurable/reranker",
            "retrieval:",
            "  candidates: 40",
            "  top_k: 10",
            "  rrf_k: 60",
        ]),
        encoding="utf-8",
    )
    spec = ExperimentSpec.load(config)
    assert spec.generator.model == "small/model-a"
    assert spec.dense_model == "configurable/dense"
    assert spec.presets == RANKIFY_PRESETS
    first = spec.generator.fingerprint()
    second = GenerationSpec.from_dict({
        "backend": "huggingface", "model": "small/model-b", "temperature": 0,
    }).fingerprint()
    assert first != second


def test_worker_exposes_the_controlled_seven_presets():
    assert tuple(PRESETS) == RANKIFY_PRESETS


def test_rankify_generation_is_greedy_and_bounded():
    api = {"generator": {"backend": "openai-compatible", "max_tokens": 123}}
    local = {"generator": {"backend": "huggingface", "max_tokens": 123}}
    assert _generation_kwargs(api) == {"temperature": 0, "max_tokens": 123}
    assert _generation_kwargs(local) == {"do_sample": False, "max_new_tokens": 123}


def test_rankify_capability_probe_and_openai_compatible_configuration(monkeypatch):
    generator_class = _fake_rankify(monkeypatch)
    assert _capabilities()["rankify_version"] == "0.1.4"
    monkeypatch.setenv("LLM_API_KEY", "runtime-only-secret")
    _generator({"generator": {
        "backend": "openai-compatible",
        "model": "openrouter/free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "LLM_API_KEY",
        "max_tokens": 768,
    }}, "basic-rag", "cuda")
    assert generator_class.last_kwargs == {
        "method": "basic-rag",
        "model_name": "openrouter/free",
        "backend": "openai",
        "api_keys": ["runtime-only-secret"],
        "base_url": "https://openrouter.ai/api/v1",
    }
    _generator({"generator": {
        "backend": "openai-compatible", "model": "openrouter/free",
        "base_url": "https://openrouter.ai/api/v1", "api_key_env": "LLM_API_KEY",
    }}, "in-context-ralm", "cuda")
    assert generator_class.last_kwargs["method"] == "basic-rag"
    assert _custom_prompt("in-context-ralm") == "native RALM prompt"


def test_rankify_question_payload_never_contains_references():
    question = QARecord(
        "q1", "its", "id", "Pertanyaan?", ["jawaban rahasia"], "test",
        source_ids=["doc-1"], metadata={"category": "akademik"},
    )
    payload = _question_payload(question)
    assert "references" not in payload
    assert "jawaban rahasia" not in json.dumps(payload, ensure_ascii=False)


def test_api_secret_is_not_serialized(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "do-not-persist")
    spec = GenerationSpec.from_dict({
        "backend": "openai-compatible", "model": "provider/model",
        "base_url": "https://example.test/v1", "api_key_env": "LLM_API_KEY",
        "temperature": 0,
    })
    assert "do-not-persist" not in json.dumps(spec.public_dict())


def test_worker_rows_join_references_only_after_generation():
    question = QARecord(
        "q1", "its", "id", "Pertanyaan?", ["jawaban benar"], "test",
        source_ids=["doc-1"], metadata={"relevance_label": "weak/category-based"},
    )
    spec = SimpleNamespace(
        generator=GenerationSpec.from_dict({
            "backend": "huggingface", "model": "small/model", "temperature": 0,
        }),
    )
    rows = _convert_worker_rows([{
        "id": "q1", "prediction": "jawaban benar", "status": "ok", "latency_ms": 1,
        "context_length": 10, "diagnostics": {},
        "evidence": [{
            "source_id": "doc-1", "chunk_id": "chunk-1", "text": "context",
            "score": 0.9, "metadata": {},
        }],
    }], [question], "rankify_bm25_basic", spec, "corpus")
    assert rows[0]["references"] == ["jawaban benar"]
    assert rows[0]["diagnostics"]["relevant_ranks"] == [1]


def test_comparison_rejects_decoding_or_corpus_mismatch(tmp_path):
    common = {
        "id": "q1", "dataset": "its", "language": "id", "split": "test",
        "question": "q", "references": ["a b c d"], "prediction": "a b c d",
        "status": "ok", "available": True, "model_fingerprint": "model",
        "gold_source_ids": [], "diagnostics": {}, "latency_ms": 1,
    }
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(baseline, [{**common, "technique": "dillema", "corpus_fingerprint": "one", "decoding_config": {"max_tokens": 10}}])
    write_jsonl(candidate, [{**common, "technique": "rankify_bm25_basic", "corpus_fingerprint": "two", "decoding_config": {"max_tokens": 10}}])
    with pytest.raises(ValueError, match="fingerprints"):
        run_compare_techniques(baseline, candidate, tmp_path / "out")
