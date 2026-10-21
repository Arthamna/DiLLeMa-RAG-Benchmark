"""Behavior and process-boundary checks for the technique separation."""

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import QdrantClient

from rag_benchmark import benchmark
from rag_benchmark.benchmark import ExperimentSpec
from rag_benchmark.shared.api_llm import APIModelConfig, OpenAICompatibleLLM
from rag_benchmark.shared.chunking import DocumentChunker
from rag_benchmark.shared.config import ChunkingConfig, GenerationSpec
from rag_benchmark.shared.datasets import DatasetBundle
from rag_benchmark.shared.evaluation import evaluate_questions
from rag_benchmark.shared.local_llm import HFModelConfig, LocalLLM
from rag_benchmark.shared.metrics import read_predictions, write_jsonl
from rag_benchmark.shared.records import DocumentRecord, QARecord, RetrievedEvidence, TechniqueResult
from rag_benchmark.techniques.dillema.augmentation import pack_parent_pages
from rag_benchmark.techniques.dillema.config import PipelineConfig
from rag_benchmark.techniques.dillema.generation import DiLLeMaGenerator, base_prompt
from rag_benchmark.techniques.dillema import runner as dillema_runner
from rag_benchmark.techniques.dillema.store import QdrantStore
from rag_benchmark.techniques.rankify.config import RANKIFY_PRESETS
from rag_benchmark.techniques.rankify.pipeline import _export_corpora, _rankify_index_root
from rag_benchmark_rankify.retrieval import _context_values, _rrf

ROOT = Path(__file__).resolve().parents[1]


def _spec(source=Path("corpus"), index_root=Path("indexes")):
    return ExperimentSpec(
        "its", source, index_root, Path("python"), "dense", "reranker", 40, 10, 60,
        "all", RANKIFY_PRESETS, GenerationSpec("huggingface", "local/model", 0, 768),
    )


def _documents():
    return [DocumentRecord(
        "doc-1", "its", "id", "Panduan café",
        "BAB I ATURAN\n" + "Peserta wajib membawa kartu pada 2027. " * 90
        + "\nBAB II JADWAL\n" + "Kuliah dimulai pukul 08.00. " * 40,
        "panduan.txt",
    )]


def test_config_chunks_packing_and_prompt_match_pre_refactor_snapshot(monkeypatch):
    monkeypatch.delenv("EMBED_MODEL_NAME", raising=False)
    monkeypatch.delenv("RERANK_MODEL_NAME", raising=False)
    expected = json.loads((ROOT / "tests/fixtures/refactor_baseline.json").read_text(encoding="utf-8"))
    assert asdict(PipelineConfig()) == expected["pipeline_config"]
    assert PipelineConfig().fingerprint() == expected["pipeline_fingerprint"]
    assert _spec().index_fingerprint() == expected["index_fingerprint"]
    assert replace(_spec(), evaluation_split="test").index_fingerprint() == expected["index_fingerprint"]
    chunks = DocumentChunker().chunk_documents(_documents())
    encoded = json.dumps([asdict(c) for c in chunks], ensure_ascii=False, sort_keys=True).encode()
    assert hashlib.sha256(encoded).hexdigest() == expected["chunks_digest"]
    assert [c.id for c in chunks] == expected["chunk_ids"]
    packed = pack_parent_pages([
        RetrievedEvidence("a", "1", "bukti pertama", .9, {"parent_id": "a", "parent_window": "bukti pertama dan konteks"}),
        RetrievedEvidence("b", "2", "lainnya", .8, {"parent_id": "b"}),
        RetrievedEvidence("a", "3", "bukti terakhir", .7, {"parent_id": "a"}),
    ], 1)
    assert packed.text == expected["packed_text"]
    assert [e.chunk_id for e in packed.evidence] == expected["packed_chunk_ids"]
    assert base_prompt("Apa aturannya?", "Bukti café: wajib membawa kartu.") == expected["prompt"]


def test_custom_chunking_is_shared_without_changing_export_schema(tmp_path):
    config = ChunkingConfig(chunk_size=400, chunk_overlap=60, min_chunk_chars=30)
    expected = DocumentChunker(**asdict(config)).chunk_documents(_documents())
    bundle = DatasetBundle(_documents(), [])
    spec = _spec(index_root=tmp_path)
    exported = _export_corpora(spec, bundle, config)
    rows = read_predictions(exported["id"])
    assert [row["id"] for row in rows] == [chunk.id for chunk in expected]
    assert [row["text"] for row in rows] == [chunk.text for chunk in expected]
    assert set(rows[0]) == {"id", "title", "text", "source_id", "source_path", "metadata"}
    assert exported["id"] == _rankify_index_root(spec) / "id/corpus.jsonl"


def test_parent_expansion_preserves_oversized_matched_leaf():
    leaf = "x" * 5500 + "jawaban penting"
    packed = pack_parent_pages([
        RetrievedEvidence("a", "1", leaf, .9, {"parent_window": "unrelated short window"}),
    ], 1)
    assert leaf in packed.text
    assert "unrelated short window" not in packed.text


def test_local_runtime_receives_the_unchanged_dillema_prompt_and_decoding():
    captured = {}
    parameter = SimpleNamespace(device="cpu", requires_grad=True)

    class Tokens(dict):
        input_ids = SimpleNamespace(shape=(1, 3))

        def to(self, device):
            captured["device"] = device
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token = "eos"

        def __call__(self, text, **kwargs):
            captured.update(text=text, tokenize=kwargs)
            return Tokens(input_ids=[1, 2, 3])

        def decode(self, values, **kwargs):
            captured.update(decoded=list(values), decode=kwargs)
            return " answer "

    class Model:
        def parameters(self):
            return iter([parameter])

        def eval(self):
            pass

        def generate(self, **kwargs):
            captured["generation"] = kwargs
            return np.array([[1, 2, 3, 4, 5]])

    runtime = LocalLLM(HFModelConfig("local", quantization="none", max_new_tokens=123), Model(), Tokenizer())
    assert DiLLeMaGenerator(runtime).generate("question", "context") == "answer"
    assert captured["text"] == base_prompt("question", "context")
    assert captured["tokenize"] == {
        "return_tensors": "pt", "truncation": True, "max_length": 4096, "add_special_tokens": True,
    }
    assert captured["generation"] == {
        "input_ids": [1, 2, 3], "do_sample": False, "max_new_tokens": 123, "pad_token_id": 0,
    }
    assert captured["decoded"] == [4, 5]
    assert parameter.requires_grad is False


def test_rankify_rrf_and_context_selection_preserve_order_and_scores():
    def context(identifier):
        return SimpleNamespace(id=identifier, score=0)

    values = _rrf([context("a"), context("b")], [context("b"), context("c")], 60, 3)
    assert [item.id for item in values] == ["b", "a", "c"]
    assert [item.score for item in values] == pytest.approx([1 / 62 + 1 / 61, 1 / 61, 1 / 62])
    document = SimpleNamespace(contexts=[context("original")], reorder_contexts=values)
    assert _context_values(document, 2) == values[:2]


def test_dillema_runner_loads_manifest_and_preserves_output_contract(tmp_path, monkeypatch):
    class Embedder:
        vector_size = 3

        def __init__(self, *args, **kwargs):
            pass

        def encode_passages(self, texts):
            return np.tile([1., 0., 0.], (len(texts), 1))

        def encode_query(self, text):
            return np.array([1., 0., 0.])

    class Reranker:
        def __init__(self, *args, **kwargs):
            pass

        def score(self, question, passages):
            return [.9] * len(passages)

    captured = []

    def complete(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])

    runtime = OpenAICompatibleLLM(
        APIModelConfig("https://example.test/v1", "runtime-secret", "model", "model"),
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete))),
    )
    bundle = DatasetBundle(_documents(), [
        QARecord("train", "its", "id", "Apa aturannya?", ["reference-only-sentinel"], "train"),
        QARecord("missing", "its", "id", "Pertanyaan?", ["reference-only-sentinel"], "test", available=False),
    ])
    monkeypatch.setenv("RAG_DEVICE", "cpu")
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setattr(dillema_runner, "load_dataset", lambda *args: bundle)
    monkeypatch.setattr(dillema_runner, "E5Embedding", Embedder)
    monkeypatch.setattr(dillema_runner, "CrossEncoderReranker", Reranker)
    client = QdrantClient(location=":memory:")
    closed = []

    class MemoryStore(QdrantStore):
        def __init__(self, target):
            super().__init__(target, client=client)

        def close(self):
            closed.append(self.index_root)

    monkeypatch.setattr(dillema_runner, "QdrantStore", MemoryStore)
    index = tmp_path / "indexes"
    indexed = dillema_runner.run_index("dillema", "its", tmp_path, index)
    assert indexed["schema_version"] == 2
    assert indexed["collections"]["its"]["points"] > 0
    output = tmp_path / "results"
    evaluated = dillema_runner.run_evaluate(
        "dillema", "its", index, output, runtime=runtime, evaluation_split="all",
    )
    assert evaluated["index_manifest"] == indexed
    assert evaluated["schema_version"] == 2
    assert evaluated["scheduled_questions"] == evaluated["prediction_records"] == 2
    rows = read_predictions(output / "predictions.jsonl")
    assert [row["status"] for row in rows] == ["ok", "missing_context"]
    assert rows[0]["references"] == ["reference-only-sentinel"]
    assert len(captured) == 1
    assert "reference-only-sentinel" not in json.dumps(captured)
    assert captured[0]["messages"][0]["content"].startswith("Answer using only the supplied evidence.")
    assert "runtime-secret" not in (output / "config.json").read_text()
    assert set(evaluated["files"]) == {path.name for path in output.iterdir()}
    assert closed == [index / "dillema/its", index / "dillema/its"]
    client.close()


@pytest.mark.parametrize("failed_preset", [None, RANKIFY_PRESETS[3]])
def test_benchmark_keeps_order_outputs_and_worker_isolation(tmp_path, monkeypatch, failed_preset):
    spec = _spec(source=tmp_path, index_root=tmp_path / "indexes")
    questions = [
        QARecord("train", "its", "id", "training question", ["private reference train"], "train"),
        QARecord("test", "its", "id", "test question", ["private reference test"], "test"),
        QARecord("missing", "its", "id", "missing question", ["private reference missing"], "test", available=False),
    ]
    bundle = DatasetBundle(_documents(), questions)
    calls = []

    class Worker:
        def __init__(self, python):
            assert python == spec.rankify_python

        def invoke(self, request):
            assert request["schema_version"] == 1
            operation = request["operation"]
            calls.append(request.get("preset", operation))
            assert "private reference" not in json.dumps(request)
            if operation == "evaluate":
                rows = []
                for batch in request["batches"]:
                    payload = Path(batch["questions_path"]).read_text(encoding="utf-8")
                    assert "private reference" not in payload
                    for question in map(json.loads, payload.splitlines()):
                        assert "references" not in question
                        rows.append({
                            **question, "prediction": "answer", "evidence": [], "context_length": 0,
                            "latency_ms": 1, "status": "ok" if question["available"] else "missing_context",
                        })
                if request["preset"] == failed_preset:
                    raise RuntimeError("fixture worker failure")
                write_jsonl(request["output_path"], rows)
            return {"status": "ok", "operation": operation}

    def baseline_index(*args, **kwargs):
        return {"schema_version": 2}

    def baseline_evaluate(technique, dataset, index_root, output, source, **kwargs):
        assert kwargs["evaluation_split"] == "all"
        runner = SimpleNamespace(answer=lambda *args: TechniqueResult("answer", "evidence", []))
        rows = evaluate_questions(
            questions, runner, dataset, technique, kwargs["model_fingerprint"],
            benchmark._corpus_fingerprint(bundle.documents), kwargs["decoding_config"],
        )
        output.mkdir(parents=True)
        write_jsonl(output / "predictions.jsonl", [row.to_dict() for row in rows])

    monkeypatch.setattr(benchmark.ExperimentSpec, "load", lambda path: spec)
    monkeypatch.setattr(benchmark, "load_dataset", lambda *args: bundle)
    monkeypatch.setattr(benchmark, "RankifyWorkerClient", Worker)
    monkeypatch.setattr(benchmark, "run_index", baseline_index)
    monkeypatch.setattr(benchmark, "run_evaluate", baseline_evaluate)
    monkeypatch.setattr(GenerationSpec, "runtime", lambda self: object())
    monkeypatch.setattr(benchmark, "_release_gpu", lambda: None)
    output = tmp_path / "run"
    manifest = benchmark.run_benchmark(tmp_path / "experiment.yaml", "all", output)
    assert calls == ["capabilities", "index", *RANKIFY_PRESETS]
    assert manifest["schema_version"] == 1
    assert list(manifest["result"]["reports"]) == ["dillema", *RANKIFY_PRESETS]
    for preset in RANKIFY_PRESETS:
        if preset == failed_preset:
            assert manifest["result"]["reports"][preset]["status"] == "failed"
            continue
        rows = read_predictions(output / preset / "predictions.jsonl")
        assert [row["id"] for row in rows] == ["train", "test", "missing"]
        assert rows[0]["references"] == ["private reference train"]
        assert rows[2]["status"] == "missing_context"
        assert rows[0]["decoding_config"] == spec.generator.decoding_dict()
    assert (output / "leaderboard.csv").is_file()
    assert (output / "leaderboard.json").is_file()


def test_shared_and_techniques_obey_import_boundaries():
    root = ROOT / "src/rag_benchmark"
    for subtree, forbidden in (
        ("shared", ("rag_benchmark.techniques",)),
        ("techniques/dillema", ("rag_benchmark.techniques.rankify", "rag_benchmark.techniques.qwen3")),
        ("techniques/rankify", ("rag_benchmark.techniques.dillema", "rag_benchmark.techniques.qwen3")),
        ("techniques/qwen3", ("rag_benchmark.techniques.dillema", "rag_benchmark.techniques.rankify")),
    ):
        for path in (root / subtree).rglob("*.py"):
            if "worker" in path.relative_to(root).parts:
                continue
            package = ".".join(("rag_benchmark", *path.relative_to(root).parent.parts))
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    name = "." * node.level + (node.module or "")
                    name = importlib.util.resolve_name(name, package)
                    assert not name.startswith(forbidden), (path, name)
                elif isinstance(node, ast.Import):
                    assert all(not alias.name.startswith(forbidden) for alias in node.names), path


def test_imports_do_not_load_models_or_rankify_in_a_fresh_process():
    code = '''
import builtins
original_import = builtins.__import__
blocked = {'torch', 'transformers', 'sentence_transformers', 'rankify', 'rag_benchmark_rankify', 'vllm'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('Unexpected runtime import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
import rag_benchmark.shared.config
import rag_benchmark.shared.chunking
import rag_benchmark.shared.evaluation
import rag_benchmark.techniques.dillema.runner
import rag_benchmark.techniques.rankify.client
import rag_benchmark.techniques.rankify.pipeline
import rag_benchmark.benchmark
import rag_benchmark.cli
'''
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), *sys.path])
    completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
