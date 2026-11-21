from __future__ import annotations

import asyncio
import json
import io
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml
from qdrant_client import QdrantClient

from rag_benchmark import benchmark
from rag_benchmark.shared.comparison import run_compare_techniques
from rag_benchmark.shared.config import GenerationSpec
from rag_benchmark.shared.datasets import DatasetBundle
from rag_benchmark.shared.metrics import read_predictions, write_jsonl
from rag_benchmark.shared.records import DocumentRecord, QARecord
from rag_benchmark.techniques.rank_zephyr import runner, worker
from rag_benchmark.techniques.rank_zephyr.config import VARIANTS


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename, dataset, split", [
    ("rankify-pipe-retro-90-its.yaml", "its", "test"),
    ("rankify-pipe-retro-90-uniqa.yaml", "uniqa", "test"),
])
def test_preset_is_opt_in_and_keeps_generator_separate(monkeypatch, tmp_path, filename, dataset, split):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv("HF_MODEL_NAME", str(tmp_path))
    path = ROOT / "configs/experiments" / filename
    spec = benchmark.ExperimentSpec.load(path, "rank_zephyr")
    assert (spec.dataset, spec.evaluation_split, spec.candidates, spec.top_k) == (dataset, split, 40, 10)
    assert spec.generator.model == str(tmp_path)
    assert spec.rank_zephyr.worker_python.endswith(str(Path(".venv-rank-llm/bin/python")))
    config = spec.rank_zephyr
    assert config.index_fingerprint("corpus", "commit") != config.index_fingerprint("corpus", "new-commit")
    assert config.index_fingerprint("corpus", "commit") == replace(
        config, reranker_snapshot="/another/snapshot",
    ).index_fingerprint("corpus", "commit")
    assert benchmark.ExperimentSpec.load(path, "dillema").rank_zephyr is None
    with pytest.raises(ValueError, match="not enabled"):
        benchmark.ExperimentSpec.load(ROOT / "configs/experiments/uniqa.yaml", "rank_zephyr")


@pytest.mark.parametrize("count, expected_windows", [(0, 0), (7, 1), (21, 2), (40, 3), (100, 9)])
def test_worker_uses_upstream_permutation_and_checks_windows(monkeypatch, count, expected_windows):
    data = ModuleType("rank_llm.data")
    data.Candidate = data.Query = data.Request = SimpleNamespace
    monkeypatch.setitem(sys.modules, "rank_llm.data", data)
    row = {
        "id": "qa", "question": "Apa jadwalnya?",
        "candidates": [{"id": str(i), "text": f"Teks {i}", "score": 1.0} for i in range(count)],
    }
    history = [
        SimpleNamespace(response=" > ".join(f"[{i}]" for i in range(1, n + 1)), token_usage={"total_tokens": 30})
        for n in worker._window_lengths(count)
    ]

    class Ranker:
        def get_num_tokens(self, question):
            assert question == row["question"]
            return 3

        async def rerank_batch_async(self, requests, **kwargs):
            request, = requests
            assert request.query.text == row["question"]
            assert request.candidates[0].doc == {"text": "Teks 0"}
            assert kwargs == {
                "rank_start": 0, "rank_end": count, "top_k_retrieve": count,
                "shuffle_candidates": False, "populate_invocations_history": True,
            }
            return [SimpleNamespace(
                query=request.query, candidates=request.candidates[::-1], invocations_history=history,
            )]

    result = asyncio.run(worker._rank(Ranker(), row))
    assert result["ranked_ids"] == [item["id"] for item in reversed(row["candidates"])]
    assert result["windows"] == expected_windows
    assert result["repaired_windows"] == 0
    if count:
        history[0].response = "[1] > [1]"
        repaired = asyncio.run(worker._rank(Ranker(), row))
        assert repaired["repaired_windows"] == 1
        assert repaired["repairs"] == [{"window": 1, "response": "[1] > [1]"}]
        history[0].token_usage = {}
        with pytest.raises(RuntimeError, match="vLLM request failed"):
            asyncio.run(worker._rank(Ranker(), row))
        history.pop()
        with pytest.raises(RuntimeError, match="expected windows"):
            asyncio.run(worker._rank(Ranker(), row))


@pytest.mark.parametrize("dataset, languages, backend", [
    ("its", ("id",), "huggingface"),
    ("uniqa", ("en", "it"), "openai-compatible"),
])
def test_benchmark_pair_same_candidates_and_one_record_per_question(
    tmp_path, monkeypatch, dataset, languages, backend,
):
    # Exercise the real chunker, Qdrant, shared evaluator and benchmark selection.
    config = yaml.safe_load((ROOT / "configs/experiments/rankify-pipe-retro-90-its.yaml").read_text())
    snapshot = tmp_path / "answer-model"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    generator = {
        "backend": backend, "model": str(snapshot) if backend == "huggingface" else "remote-reader",
        "temperature": 0, "max_tokens": 128,
    }
    if backend == "openai-compatible":
        generator.update(base_url="https://api.invalid", api_key_env="TEST_GENERATOR_KEY")
    key_env = generator.get("api_key_env", "LLM_API_KEY")
    monkeypatch.setenv(key_env, "PRIVATE-GENERATOR-KEY")
    interrupted = dataset == "uniqa"
    monkeypatch.setenv("RAG_DEVICE", "cpu")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform="linux", exc_info=sys.exc_info))
    config.update(dataset=dataset, generator=generator, source=str(tmp_path), index_root=str(tmp_path / "indexes"))
    python = tmp_path / "worker-python"
    python.touch()
    config["rank_zephyr"]["worker_python"] = str(python)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    documents = [
        DocumentRecord(
            f"{language}-{i}", dataset, language, f"Document {i}",
            (f"{language} source {i} campus rules for registration and library access. " * 4).strip(),
            f"{language}-{i}.txt",
        )
        for language in languages for i in range(100)
    ]
    questions = [
        QARecord(
            f"{language}-q{i}", dataset, language, f"Public {language} question {i}?",
            ["PRIVATE-REFERENCE"], "test", i != 2, [f"{language}-0", "PRIVATE-GOLD-ID"],
            {"relevance_label": "weak/category-based" if dataset == "its" else "source"},
        )
        for language in languages for i in range(3)
    ]
    bundle = DatasetBundle(documents, questions)
    monkeypatch.setattr(benchmark, "load_dataset", lambda *_: bundle)
    client = QdrantClient(":memory:")
    class MemoryStore(runner.DenseStore):
        def __init__(self, _path, namespace):
            self.client = client
            self.namespace = namespace

        def close(self):
            pass

    monkeypatch.setattr(runner, "DenseStore", MemoryStore)
    stages = []

    class Embedding:
        dimension = 2
        revision = "resolved-embedding-commit"

        def __init__(self, config, device):
            assert device == "cpu"

        def encode(self, texts):
            assert all("PRIVATE" not in text for text in texts)
            return np.array([[1.0, 0.0] for _ in texts])

    def fake_worker(command, env, **kwargs):
        assert key_env not in env
        assert command[1:3] == ["-I", "-u"]
        assert kwargs["check"]
        stages.append("worker")
        def file(flag):
            return Path(command[command.index(flag) + 1])
        requests = read_predictions(file("--input"))
        assert len(requests) == 2 * len(languages)
        assert all(len(row["candidates"]) == 40 for row in requests)
        assert "PRIVATE" not in file("--input").read_text(encoding="utf-8")
        write_jsonl(file("--output"), [
            {"id": row["id"], "ranked_ids": [c["id"] for c in reversed(row["candidates"])],
             "windows": 3, "repaired_windows": 1, "rerank_latency_ms": 3.0}
            for row in requests if not interrupted or row["id"] != f"{languages[0]}-q1"
        ])
        diagnostics = {"status": "failed", "error": "CUDA worker interrupted"} if interrupted else {
            "status": "complete",
        }
        file("--diagnostics").write_text(json.dumps(diagnostics))
        if interrupted:
            raise subprocess.CalledProcessError(1, command)

    class Generator:
        def generate_prompt(self, prompt):
            assert "PRIVATE" not in prompt
            stages.append("generate")
            return "Campus information answer"

    monkeypatch.setattr(runner, "DenseEmbedding", Embedding)
    monkeypatch.setattr(runner.subprocess, "run", fake_worker)
    monkeypatch.setattr(GenerationSpec, "runtime", lambda self: Generator())
    output = tmp_path / "run"
    report = benchmark.run_benchmark(path, "all", output, "rank_zephyr")
    index = report["result"]["index"]["rank_zephyr"]
    assert set(index["collections"]) == set(languages)
    assert len({item["name"] for item in index["collections"].values()}) == len(languages)
    assert stages[0] == "worker"
    assert stages.count("generate") == 4 * len(languages) - int(interrupted)
    files = [output / variant / "predictions.jsonl" for variant in VARIANTS]
    dense, reranked = map(read_predictions, files)
    assert [row["id"] for row in dense] == [qa.id for qa in questions]
    assert [row["id"] for row in reranked] == [qa.id for qa in questions]
    assert [row["diagnostics"]["candidate_pool_hash"] for row in dense] == [
        row["diagnostics"]["candidate_pool_hash"] for row in reranked
    ]
    assert reranked[0]["diagnostics"]["rerank_windows"] == 3
    assert reranked[0]["diagnostics"]["ranked_chunk_ids"] != dense[0]["diagnostics"]["ranked_chunk_ids"]
    if interrupted:
        assert reranked[1]["status"] == "error"
        assert reranked[1]["prediction"] == ""
        assert "CUDA worker interrupted" in reranked[1]["error"]
    else:
        assert reranked[1]["status"] == "ok"
    assert reranked[2]["status"] == "missing_context"
    assert (output / VARIANTS[1] / "worker-diagnostics.json").is_file()
    dense_manifest = json.loads((output / VARIANTS[0] / "manifest.json").read_text())
    rerank_manifest = json.loads((output / VARIANTS[1] / "manifest.json").read_text())
    assert dense_manifest["model_fingerprint"] == rerank_manifest["model_fingerprint"]
    assert dense_manifest["decoding_config"] == rerank_manifest["decoding_config"]
    leaderboard = json.loads((output / "leaderboard.json").read_text())
    assert {row["technique"] for row in leaderboard} == set(VARIANTS[:1] if interrupted else VARIANTS)
    run_compare_techniques(*files, tmp_path / "comparison")
    client.close()


def test_incomplete_local_snapshot_fails_without_download(tmp_path):
    with pytest.raises(ValueError, match="config/tokenizer"):
        worker._snapshot("offline", "pinned", str(tmp_path))


@pytest.mark.parametrize("failure", [None, "startup", "rerank"])
def test_vllm_server_is_stopped_on_success_and_failure(tmp_path, monkeypatch, failure):
    stopped = []
    process = SimpleNamespace(
        poll=lambda: 1 if failure == "startup" else None,
        returncode=1 if failure == "startup" else None,
    )
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(worker, "_stop_server", stopped.append)
    snapshot = tmp_path / "model"
    monkeypatch.setattr(worker, "build_opener", lambda *_: SimpleNamespace(
        open=lambda *args, **kwargs: io.StringIO(json.dumps({"data": [{"id": str(snapshot)}]})),
    ))
    def run():
        with worker._server(snapshot, tmp_path / "vllm.log"):
            if failure == "rerank":
                raise RuntimeError("rerank failed")
    if failure:
        with pytest.raises(RuntimeError):
            run()
    else:
        run()
    assert stopped == [process]
