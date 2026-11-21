from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rag_benchmark import benchmark
from rag_benchmark.shared.comparison import run_compare_techniques
from rag_benchmark.shared.datasets import DatasetBundle
from rag_benchmark.shared.metrics import summarize, write_jsonl
from rag_benchmark.shared.records import QARecord
from rag_benchmark.techniques.bge_m3.config import VARIANTS as BGE_VARIANTS
from rag_benchmark.techniques.self_rag.config import VARIANTS as SELF_VARIANTS
from rag_benchmark.techniques.rank_zephyr.config import VARIANTS as RZ_VARIANTS

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("dillema", "bge_m3", "self_rag", "rank_zephyr")
VARIANTS = ("dillema", *BGE_VARIANTS, *SELF_VARIANTS, *RZ_VARIANTS)


def preset(dataset="its"):
    return ROOT / "configs/experiments" / f"rankify-pipe-retro-90-{dataset}.yaml"


def temporary_config(tmp_path, **overrides):
    config = yaml.safe_load(preset().read_text())
    config.update(source=str(tmp_path), index_root=str(tmp_path / "indexes"))
    config.update(overrides)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("dataset", ["its", "uniqa"])
def test_presets_keep_128_tokens_despite_legacy_environment(dataset, tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv("HF_MODEL_NAME", str(tmp_path))
    monkeypatch.setenv("HF_MAX_NEW_TOKENS", "768")
    monkeypatch.setenv("LLM_MAX_TOKENS", "768")
    spec = benchmark.ExperimentSpec.load(preset(dataset))
    assert spec.techniques == GROUPS
    assert (spec.evaluation_split, spec.candidates, spec.top_k) == ("test", 40, 10)
    assert spec.generator.max_tokens == spec.self_rag.max_new_tokens == 128
    assert spec.generator.profile == "qwen"
    assert spec.generator.quantization == "4bit"
    assert spec.self_rag.embedding_batch_size == 8
    assert spec.services is spec.retro is spec.piperag is None
    assert spec.index_root == ROOT / "artifacts/indexes/pipe-retro-90"
    public = spec.public_dict()
    assert public["retrieval"] == {"candidates": 40, "top_k": 10}
    assert all(public[name]["enabled"] for name in GROUPS[1:])


def test_cli_selection_ignores_invalid_unselected_configs(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_MODEL_NAME", raising=False)
    path = temporary_config(
        tmp_path, generator={"active": "missing"},
        bge_m3={"enabled": True, "model": "./missing"},
        rank_zephyr={"enabled": True}, retro={"invalid": True},
        services={"invalid": True}, paper_evaluation={"unknown": True},
    )
    spec = benchmark.ExperimentSpec.load(path, "self_rag")
    assert spec.techniques == ("self_rag",)
    assert spec.generator is spec.bge_m3 is spec.rank_zephyr is spec.services is None


def test_api_profile_and_disabled_selected_technique(tmp_path, monkeypatch):
    raw = yaml.safe_load(preset().read_text())
    raw["generator"]["active"] = "openai"
    monkeypatch.setenv("LLM_MODEL", "reader")
    monkeypatch.setenv("LLM_MAX_TOKENS", "768")
    monkeypatch.setenv("LLM_API_KEY", "secret-not-in-manifest")
    path = temporary_config(tmp_path, generator=raw["generator"])
    spec = benchmark.ExperimentSpec.load(path, "bge_m3,rank_zephyr")
    assert spec.generator.backend == "openai-compatible"
    assert spec.generator.max_tokens == 128
    assert "secret-not-in-manifest" not in json.dumps(spec.public_dict())
    path = temporary_config(tmp_path, bge_m3={"enabled": False})
    with pytest.raises(ValueError, match="not enabled"):
        benchmark.ExperimentSpec.load(path, "bge_m3")


def test_config_selection_rejects_empty_and_unknown_groups(tmp_path):
    for selected in ([], ["typo"]):
        with pytest.raises(ValueError, match="subset"):
            benchmark.ExperimentSpec.load(temporary_config(tmp_path, techniques=selected))


def test_configuration_import_does_not_load_local_runners_or_model_libraries():
    code = """
import sys
from rag_benchmark.benchmark import ExperimentSpec
for name in ('bge_m3', 'self_rag', 'rank_zephyr'):
    assert f'rag_benchmark.techniques.{name}.runner' not in sys.modules
for name in ('FlagEmbedding', 'rank_llm', 'vllm'):
    assert name not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def prediction(technique, model="reader", tokens=128, qid="q1", language="id", dataset="its"):
    return {
        "id": qid, "dataset": dataset, "language": language, "split": "test",
        "technique": technique, "model_fingerprint": model, "corpus_fingerprint": "corpus",
        "decoding_config": {"temperature": 0, "max_tokens": tokens},
        "prediction": "answer", "references": ["answer"], "latency_ms": 1,
        "status": "ok", "available": True, "diagnostics": {},
    }


@pytest.mark.parametrize("model,tokens,expected", [
    ("reader", 128, "controlled_same_checkpoint"),
    ("self-rag", 128, "cross_model"),
    ("reader", 100, "cross_model"),
])
def test_comparison_uses_generator_and_decoding(tmp_path, model, tokens, expected):
    left, right = tmp_path / "left.jsonl", tmp_path / "right.jsonl"
    write_jsonl(left, [prediction("rank_zephyr_dense")])
    write_jsonl(right, [prediction("rank_zephyr_rerank", model, tokens)])
    result = run_compare_techniques(left, right, tmp_path / "comparison")
    assert result["comparability"] == expected
    assert (result["warning"] is None) == (expected == "controlled_same_checkpoint")


@pytest.mark.parametrize("field,value", [
    ("dataset", "uniqa"), ("split", "train"), ("corpus_fingerprint", "other"), ("id", "q2"),
])
def test_comparison_still_rejects_different_evaluation_populations(tmp_path, field, value):
    left, right = tmp_path / "left.jsonl", tmp_path / "right.jsonl"
    row = prediction("dillema")
    write_jsonl(left, [row])
    write_jsonl(right, [{**row, field: value}])
    with pytest.raises(ValueError, match="identical dataset"):
        run_compare_techniques(left, right, tmp_path / "comparison")


@pytest.mark.parametrize("dataset,languages", [("its", ("id",)), ("uniqa", ("en", "it"))])
def test_all_four_groups_write_ten_outputs_with_one_index_per_group(tmp_path, monkeypatch, dataset, languages):
    path = temporary_config(tmp_path, dataset=dataset, generator={
        "backend": "openai-compatible", "model": "reader", "max_tokens": 128,
    })
    questions = [QARecord(f"q-{lang}", dataset, lang, "question", ["answer"], "test") for lang in languages]
    bundle = DatasetBundle([], questions)
    events = []
    monkeypatch.setattr(benchmark, "load_dataset", lambda *_: bundle)
    monkeypatch.setattr(benchmark, "_gpu_lock_root", lambda: tmp_path / "lock")
    monkeypatch.setattr(benchmark, "release_cuda_memory", lambda *_: None)

    def reports(output, variants):
        result = {}
        for variant in variants:
            rows = [prediction(variant, qid=q.id, language=q.language, dataset=dataset) for q in questions]
            folder = output / variant
            folder.mkdir(parents=True, exist_ok=True)
            write_jsonl(folder / "predictions.jsonl", rows)
            result[variant] = summarize(rows)
        return result

    def index_dillema(*args, **kwargs):
        assert kwargs["pipeline_config"].retrieval_candidates == 40
        events.append(("index", "dillema"))
        return {"status": "complete"}

    def evaluate_dillema(_technique, _dataset, _index, output, *_args, **kwargs):
        assert "runtime_factory" in kwargs and "runtime" not in kwargs
        assert kwargs["evaluation_split"] == "test"
        events.append(("evaluate", "dillema"))
        reports(output.parent, ("dillema",))

    def local_runner(group):
        variants = {"bge_m3": BGE_VARIANTS, "self_rag": SELF_VARIANTS, "rank_zephyr": RZ_VARIANTS}[group]

        def index(spec, data):
            assert data is bundle
            events.append(("index", group))
            return {"status": "complete"}

        def evaluate(spec, data, output):
            assert data is bundle
            events.append(("evaluate", group))
            return reports(output, variants)

        return SimpleNamespace(run_index=index, run_evaluate=evaluate)

    def forbidden(*args, **kwargs):
        pytest.fail("RETRO services must not run")

    monkeypatch.setattr(benchmark, "index_dillema", index_dillema)
    monkeypatch.setattr(benchmark, "evaluate_dillema", evaluate_dillema)
    monkeypatch.setattr(benchmark, "_local_runner", local_runner)
    monkeypatch.setattr(benchmark, "index_retro", forbidden)
    monkeypatch.setattr(benchmark, "evaluate_retro", forbidden)
    monkeypatch.setattr(benchmark, "_unload_remote_inference", forbidden)
    output = tmp_path / "run"
    manifest = benchmark.run_benchmark(path, "all", output)
    assert events == [(stage, group) for stage in ("index", "evaluate") for group in GROUPS]
    assert set(manifest["result"]["reports"]) == set(VARIANTS)
    assert manifest["comparability"]["self_rag_vs_shared_generator"] == "cross-model"
    leaderboard = json.loads((output / "leaderboard.json").read_text())
    assert len(leaderboard) == 10 * len(languages)
    assert {row["technique"] for row in leaderboard} == set(VARIANTS)


def test_gpu_lock_covers_different_index_namespaces(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "_gpu_lock_root", lambda: tmp_path / "shared-lock")
    path = temporary_config(tmp_path, techniques=["self_rag"])
    with benchmark._gpu_lease(benchmark._gpu_lock_root(), tmp_path / "first"):
        with pytest.raises(RuntimeError, match="already active"):
            benchmark.run_benchmark(path, "index", tmp_path / "second")
