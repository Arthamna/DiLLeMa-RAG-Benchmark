from contextlib import nullcontext
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from qdrant_client import QdrantClient

from rag_benchmark import benchmark
from rag_benchmark.benchmark import ExperimentSpec
from rag_benchmark.shared.config import ChunkingConfig, GenerationSpec
from rag_benchmark.shared.datasets import DatasetBundle
from rag_benchmark.shared.device import CUDAStageError
from rag_benchmark.shared.metrics import read_predictions, summarize, write_jsonl
from rag_benchmark.shared.records import DocumentRecord, QARecord, RetrievedEvidence
from rag_benchmark.techniques.qwen3 import runner
from rag_benchmark.techniques.qwen3.config import Qwen3Config, VARIANTS
from rag_benchmark.techniques.qwen3.embedding import Qwen3Embedding, last_token_pool, model_options
from rag_benchmark.techniques.qwen3.pipeline import candidate_hash, rerank_candidates
from rag_benchmark.techniques.qwen3.reranker import Qwen3Reranker, relevance_probability
from rag_benchmark.techniques.qwen3.store import DenseStore
from rag_benchmark.techniques.rankify.config import RANKIFY_PRESETS
from rag_benchmark.techniques.bge_m3.config import BgeM3Config, VARIANTS as BGE_VARIANTS


def config():
    return Qwen3Config("fake/embedding", "fake/reranker", "Retrieve relevant passages.", embedding_dimension=32)


def test_config_local_snapshots_and_index_isolation(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    values = {"enabled": True, **config().public_dict(), "embedding_model": "./snapshot"}
    parsed = Qwen3Config.from_dict(values, tmp_path)
    assert parsed.embedding_model == str(snapshot.resolve())
    assert model_options(parsed.embedding_model, "unused") == {"local_files_only": True}
    assert model_options("org/model", "commit") == {"revision": "commit"}
    original = parsed.index_fingerprint("corpus", ChunkingConfig())
    assert original == replace(parsed, reranker_model="other/reranker", instruction="new query instruction").index_fingerprint("corpus", ChunkingConfig())
    assert original != replace(parsed, embedding_dimension=64).index_fingerprint("corpus", ChunkingConfig())
    assert original != parsed.index_fingerprint("changed corpus", ChunkingConfig())
    assert original != replace(parsed, embedding_revision="new").index_fingerprint("corpus", ChunkingConfig())
    assert Qwen3Config.from_dict({}, tmp_path) is None
    with pytest.raises(FileNotFoundError):
        Qwen3Config.from_dict({**values, "embedding_model": "./missing"}, tmp_path)


class Tensor(np.ndarray):
    @property
    def device(self):
        return "cpu"

    def float(self):
        return self.astype(np.float32)

    def sum(self, axis=None, dim=None, **kwargs):
        return super().sum(axis=dim if dim is not None else axis, **kwargs)

    def softmax(self, dim):
        values = np.exp(self - self.max(axis=dim, keepdims=True))
        return values / values.sum(axis=dim, keepdims=True)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def tensor(values):
    return np.asarray(values).view(Tensor)


def test_official_last_token_pooling_probability_and_query_instruction(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(arange=lambda size, **_: np.arange(size)))
    hidden = tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    assert last_token_pool(hidden, tensor([[0, 1, 1], [1, 1, 1]])).tolist() == [[5, 6], [11, 12]]
    assert last_token_pool(hidden, tensor([[1, 1, 0], [1, 0, 0]])).tolist() == [[3, 4], [7, 8]]
    scores = relevance_probability(tensor([[1000, 1002, -9], [7, 7, 0]]), 0, 1)
    assert scores.tolist() == pytest.approx([0.880797, 0.5])
    embedder = Qwen3Embedding.__new__(Qwen3Embedding)
    embedder.config = config()
    captured = []
    embedder._encode = lambda values: captured.append(values) or [[1, 0]]
    embedder.encode_query("Pertanyaan café?")
    embedder.encode_passages(["Dokumen tanpa instruction."])
    assert captured == [["Instruct: Retrieve relevant passages.\nQuery: Pertanyaan café?"], ["Dokumen tanpa instruction."]]


def test_embedding_forward_normalizes_after_dimension_truncation(monkeypatch):
    torch = ModuleType("torch")
    nn = ModuleType("torch.nn")
    functional = ModuleType("torch.nn.functional")
    torch.nn = nn
    nn.functional = functional
    torch.inference_mode = nullcontext
    functional.normalize = lambda values, **kwargs: values / np.linalg.norm(values, axis=1, keepdims=True)
    for name, module in (("torch", torch), ("torch.nn", nn), ("torch.nn.functional", functional)):
        monkeypatch.setitem(sys.modules, name, module)

    class Batch(dict):
        def to(self, device):
            return self

    embedder = Qwen3Embedding.__new__(Qwen3Embedding)
    embedder.config = replace(config(), embedding_batch_size=1)
    embedder.device = "cpu"
    embedder.vector_size = 2
    embedder.tokenizer = lambda texts, **kwargs: Batch(attention_mask=tensor([[1, 1]]))
    embedder.model = lambda **tokens: SimpleNamespace(last_hidden_state=tensor([[[1., 0., 0.], [3., 4., 99.]]]))
    vectors = embedder.encode_passages(["first document", "second document"])
    assert vectors.shape == (2, 2)
    np.testing.assert_allclose(vectors, [[.6, .8], [.6, .8]])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1, 1])


@pytest.mark.parametrize("device,quantization", [("cpu", "4bit"), ("cuda", "4bit"), ("cuda", "none")])
def test_reranker_loader_and_forward_use_last_token_logits(monkeypatch, device, quantization):
    observed = {}

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0

        def encode(self, text, **kwargs):
            return [3, 4]

        def convert_tokens_to_ids(self, text):
            return {"no": 0, "yes": 1}[text]

        def __call__(self, texts, **kwargs):
            observed["texts"] = texts
            observed["tokenization"] = kwargs
            return {"input_ids": [[5]] * len(texts)}

        def pad(self, tokens, **kwargs):
            observed["padded"] = tokens
            return Batch(input_ids=tokens["input_ids"])

    class Model:
        config = SimpleNamespace(_commit_hash="resolved-revision")

        def eval(self):
            return self

        def to(self, device):
            observed["device"] = device
            return self

        def __call__(self, **kwargs):
            observed["forward"] = kwargs
            return SimpleNamespace(logits=tensor([[[0., 2., -4.]]]))

    def load_model(*args, **kwargs):
        observed["load"] = kwargs
        return Model()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        bfloat16="bf16", float32="fp32", inference_mode=nullcontext,
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        BitsAndBytesConfig=lambda **kwargs: kwargs,
    ))
    monkeypatch.setattr("rag_benchmark.techniques.qwen3.reranker.synchronize_cuda", lambda *args: None)
    reranker = Qwen3Reranker(replace(config(), reranker_quantization=quantization), device)
    assert reranker.score("question", ["document"]) == pytest.approx([0.880797])
    assert observed["forward"]["logits_to_keep"] == 1
    assert observed["forward"]["use_cache"] is False
    assert observed["padded"]["input_ids"] == [[3, 4, 5, 3, 4]]
    assert "<Query>: question\n<Document>: document" in observed["texts"][0]
    if device == "cpu":
        assert reranker.quantization == "none"
        assert observed["load"]["torch_dtype"] == "fp32"
        assert "quantization_config" not in observed["load"]
    elif quantization == "4bit":
        assert observed["load"]["quantization_config"]["load_in_4bit"] is True
        assert observed["load"]["device_map"] == {"": "cuda"}


def test_rerank_preserves_candidate_pool_and_stable_ties():
    evidence = [RetrievedEvidence("doc", str(i), f"passage {i}", float(i)) for i in range(3)]
    original = candidate_hash(evidence)
    ranked = rerank_candidates("q", evidence, SimpleNamespace(score=lambda *args: [.1, .9, .9]))
    assert [item.chunk_id for item in ranked] == ["1", "2", "0"]
    assert candidate_hash(evidence) == original
    assert {item.chunk_id for item in ranked} == {item.chunk_id for item in evidence}
    with pytest.raises(RuntimeError, match="incomplete"):
        rerank_candidates("q", evidence, SimpleNamespace(score=lambda *args: []))


@pytest.fixture
def qwen_environment(tmp_path, monkeypatch):
    active, events, prompts = set(), [], []
    client = QdrantClient(location=":memory:")

    class Embedder:
        vector_size = 32
        revision = "embedding-commit"

        def __init__(self, *args):
            assert not active
            active.add("embedding")
            events.append("load-embedding")

        def encode_passages(self, texts):
            vectors = np.zeros((len(texts), 32))
            vectors[:, 0] = 1
            return vectors

        def encode_query(self, question):
            if "retrieval-fails" in question:
                raise ValueError("retrieval fixture failure")
            vector = np.zeros(32)
            vector[0] = 1
            return vector

        def __del__(self):
            active.discard("embedding")
            events.append("unload-embedding")

    class Reranker:
        revision = "reranker-commit"
        quantization = "none"

        def __init__(self, *args):
            assert not active
            active.add("reranker")
            events.append("load-reranker")

        def score(self, question, texts):
            if "rerank-fails" in question:
                raise ValueError("reranking fixture failure")
            return [float(i + 1) / len(texts) for i in range(len(texts))]

        def __del__(self):
            active.discard("reranker")
            events.append("unload-reranker")

    class Runtime:
        def __init__(self):
            assert not active
            events.append("load-generator")

        def generate_prompt(self, prompt):
            assert not active
            prompts.append(prompt)
            if "generation-fails" in prompt:
                raise ValueError("generation fixture failure")
            return "answer café one two three"

    class Store(DenseStore):
        def __init__(self, root, namespace):
            super().__init__(root, namespace, client=client)

    monkeypatch.setattr(runner, "Qwen3Embedding", Embedder)
    monkeypatch.setattr(runner, "Qwen3Reranker", Reranker)
    monkeypatch.setattr(runner, "DenseStore", Store)
    monkeypatch.setattr(runner, "resolve_torch_device", lambda *args: "cpu")
    monkeypatch.setattr(runner, "_release_gpu", gc.collect)
    monkeypatch.setattr(GenerationSpec, "runtime", lambda self: Runtime())
    monkeypatch.delenv("RAG_DEVICE", raising=False)
    yield tmp_path, events, prompts
    client.close()


def spec_and_bundle(tmp_path, dataset, failures=False):
    languages = ["id"] if dataset == "its" else ["en", "it"]
    documents, questions = [], []
    for language in languages:
        for index in range(2):
            identifier = f"doc-{language}-{index}"
            documents.append(DocumentRecord(identifier, dataset, language, "Title", "Corpus content café. " * 20, identifier))
            questions.append(QARecord(
                f"qa-{language}-{index}", dataset, language, f"Question {language} {index}?",
                ["reference-only-sentinel"], "test", source_ids=[identifier],
            ))
    if failures:
        for name in ("retrieval-fails", "rerank-fails", "generation-fails", "missing"):
            questions.append(QARecord(name, dataset, "id", name, ["reference-only-sentinel"], "test", available=name != "missing"))
    spec = ExperimentSpec(
        dataset, tmp_path, tmp_path / "indexes", Path("python"), "dense", "reranker",
        100, 10, 60, "all", RANKIFY_PRESETS,
        GenerationSpec("huggingface", "local/generator", 0, 768), config(),
    )
    return spec, DatasetBundle(documents, questions)


@pytest.mark.parametrize("dataset", ["its", "uniqa"])
def test_qwen_smoke_all_languages_and_model_lifecycle(qwen_environment, dataset):
    tmp_path, events, prompts = qwen_environment
    spec, bundle = spec_and_bundle(tmp_path, dataset)
    manifest = runner.run_index(spec, bundle)
    assert manifest["status"] == "complete"
    assert manifest["embedding_revision"] == "embedding-commit"
    output = tmp_path / "run"
    reports = runner.run_evaluate(spec, bundle, output)
    assert set(reports) == set(VARIANTS)
    assert events[-5:] == ["load-embedding", "unload-embedding", "load-reranker", "unload-reranker", "load-generator"]
    assert all("reference-only-sentinel" not in prompt for prompt in prompts)
    pools = []
    for variant in VARIANTS:
        rows = read_predictions(output / variant / "predictions.jsonl")
        assert len(rows) == len(bundle.questions)
        assert all(row["status"] == "ok" for row in rows)
        assert all(row["diagnostics"]["gpu_fallback"] for row in rows)
        assert all(row["references"] == ["reference-only-sentinel"] for row in rows)
        assert all(row["model_fingerprint"] == spec.generator.fingerprint() for row in rows)
        pools.append([row["diagnostics"]["candidate_pool_hash"] for row in rows])
    assert pools[0] == pools[1]
    assert json.loads((output / "qwen3-ablation.json").read_text())["candidate_pools_identical"]
    from rag_benchmark.shared.comparison import run_compare_techniques

    comparison = run_compare_techniques(
        output / VARIANTS[0] / "predictions.jsonl", output / VARIANTS[1] / "predictions.jsonl",
        output / "comparison",
    )
    assert comparison["baseline_technique"] == VARIANTS[0]
    with pytest.raises(ValueError, match="incomplete"):
        bad_manifest = runner.index_root(spec, manifest["corpus_fingerprint"]) / "index-manifest.json"
        bad_manifest.write_text('{"status":"building"}')
        runner.run_evaluate(spec, bundle, output)


def test_qwen_errors_and_missing_context_stay_in_denominator(qwen_environment):
    tmp_path, _, prompts = qwen_environment
    spec, bundle = spec_and_bundle(tmp_path, "its", failures=True)
    runner.run_index(spec, bundle)
    reports = runner.run_evaluate(spec, bundle, tmp_path / "run")
    for variant in VARIANTS:
        rows = {row["id"]: row for row in read_predictions(tmp_path / "run" / variant / "predictions.jsonl")}
        assert len(rows) == reports[variant]["its"]["count"] == 6
        assert rows["missing"]["status"] == "missing_context"
        for identifier in ("retrieval-fails", "generation-fails"):
            assert rows[identifier]["prediction"] == ""
            assert rows[identifier]["status"] == "error"
        assert rows["rerank-fails"]["status"] == ("error" if variant == VARIANTS[1] else "ok")
    assert all("reference-only-sentinel" not in prompt for prompt in prompts)


def test_qwen_fatal_cuda_errors_propagate():
    with pytest.raises(CUDAStageError):
        runner._raise_if_fatal(CUDAStageError("qwen3-reranking fixture"))


def test_four_experiment_presets_enable_qwen_without_changing_generator(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setenv("HF_MODEL_NAME", str(snapshot.resolve()))
    monkeypatch.setenv("LLM_MODEL", "configured/api-model")
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs/experiments").glob("rankify-*.yaml"):
        spec = ExperimentSpec.load(path)
        assert spec.qwen3 is not None
        assert spec.bge_m3 is not None
        assert spec.candidates == 100
        assert spec.evaluation_split == ("all" if spec.dataset == "its" else "test")
        assert spec.generator.model == (str(snapshot.resolve()) if "local" in path.stem else "configured/api-model")
        assert "qwen3" in spec.public_dict()
        assert "bge_m3" in spec.public_dict()
        assert replace(spec, qwen3=replace(spec.qwen3, reranker_quantization="4bit")).fingerprint() != spec.fingerprint()


@pytest.mark.parametrize("operation", ["index", "evaluate"])
def test_qwen_cli_routes_to_its_own_runner(tmp_path, monkeypatch, capsys, operation):
    from rag_benchmark import cli

    spec, bundle = spec_and_bundle(tmp_path, "its")
    monkeypatch.setattr(ExperimentSpec, "load", lambda path: spec)
    monkeypatch.setattr("rag_benchmark.shared.datasets.load_dataset", lambda *args: bundle)
    calls = []
    monkeypatch.setattr(runner, "run_index", lambda loaded, data: calls.append((loaded, data)) or {"ok": True})
    monkeypatch.setattr(runner, "run_evaluate", lambda loaded, data, output, variants: calls.append((loaded, data, variants)) or {"ok": True})
    arguments = [operation, "--technique", VARIANTS[0], "--dataset", "its", "--config", "experiment.yaml"]
    arguments += ["--source", str(tmp_path)] if operation == "index" else ["--output", str(tmp_path / "run")]
    cli.main(arguments)
    assert calls[0][0].index_root == spec.index_root
    assert calls[0][1] is bundle
    assert json.loads(capsys.readouterr().out) == {"ok": True}


@pytest.mark.parametrize("dataset", ["its", "uniqa"])
def test_fifteen_pipeline_benchmark_smoke(qwen_environment, monkeypatch, dataset):
    tmp_path, _, _ = qwen_environment
    spec, bundle = spec_and_bundle(tmp_path, dataset)
    spec = replace(spec, bge_m3=BgeM3Config(revision="resolved-bge-commit"))
    bge_index_calls = []

    class Worker:
        def __init__(self, *args):
            pass

        def invoke(self, request):
            if request["operation"] == "evaluate":
                rows = []
                for batch in request["batches"]:
                    for item in read_predictions(batch["questions_path"]):
                        assert "references" not in item
                        rows.append({**item, "prediction": "answer", "evidence": [], "status": "ok"})
                write_jsonl(request["output_path"], rows)
            return {"status": "ok"}

    def dillema_evaluate(technique, dataset, index_root, output, source, **kwargs):
        assert kwargs["pipeline_config"].retrieval_candidates == 100
        output.mkdir(parents=True)
        write_jsonl(output / "predictions.jsonl", [{
            "id": q.id, "dataset": dataset, "language": q.language, "references": q.references,
            "prediction": "answer", "technique": technique, "status": "ok",
            "model_fingerprint": spec.generator.fingerprint(),
        } for q in bundle.questions])

    monkeypatch.setattr(ExperimentSpec, "load", lambda path: spec)
    monkeypatch.setattr(benchmark, "load_dataset", lambda *args: bundle)
    monkeypatch.setattr(benchmark, "RankifyWorkerClient", Worker)
    monkeypatch.setattr(benchmark, "run_index", lambda *args, **kwargs: {})
    monkeypatch.setattr(benchmark, "run_evaluate", dillema_evaluate)
    monkeypatch.setattr(
        benchmark, "index_bge_m3",
        lambda loaded, data: bge_index_calls.append((loaded, data)) or {"status": "complete"},
    )

    def bge_evaluate(loaded, data, output):
        reports = {}
        for variant in BGE_VARIANTS:
            target = output / variant
            target.mkdir(parents=True)
            rows = [{
                "id": q.id, "dataset": dataset, "language": q.language,
                "question": q.question, "references": q.references,
                "prediction": "answer", "technique": variant, "status": "ok",
                "available": q.available, "split": q.split, "latency_ms": 1,
                "model_fingerprint": spec.generator.fingerprint(),
                "corpus_fingerprint": "corpus", "decoding_config": {},
                "gold_source_ids": q.source_ids, "diagnostics": {"relevant_ranks": []},
            } for q in bundle.questions]
            write_jsonl(target / "predictions.jsonl", rows)
            reports[variant] = summarize(rows)
        return reports

    monkeypatch.setattr(benchmark, "evaluate_bge_m3", bge_evaluate)
    monkeypatch.setattr(benchmark, "_release_gpu", lambda **kwargs: gc.collect())
    monkeypatch.setattr(benchmark, "_log_cuda_runtime", lambda *args: None)
    result = benchmark.run_benchmark(tmp_path / "config.yaml", "all", tmp_path / "run")
    assert len(bge_index_calls) == 1
    assert len(result["result"]["reports"]) == 15
    assert len(json.loads((tmp_path / "run/leaderboard.json").read_text())) == (15 if dataset == "its" else 30)
    for technique in result["result"]["reports"]:
        assert len(read_predictions(tmp_path / "run" / technique / "predictions.jsonl")) == len(bundle.questions)
