from __future__ import annotations

import gc
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from rag_benchmark.benchmark import ExperimentSpec
from rag_benchmark import benchmark
from rag_benchmark.shared.config import ChunkingConfig, GenerationSpec
from rag_benchmark.shared.datasets import DatasetBundle
from rag_benchmark.shared.metrics import read_predictions
from rag_benchmark.shared.records import ChunkRecord, DocumentRecord, QARecord, RetrievedEvidence
from rag_benchmark.techniques.bge_m3 import runner
from rag_benchmark.techniques.bge_m3 import embedding as embedding_module
from rag_benchmark.techniques.bge_m3.config import BgeM3Config, VARIANTS
from rag_benchmark.techniques.bge_m3.embedding import BgeM3Batch, BgeM3Encoding, _sparse_vector
from rag_benchmark.techniques.bge_m3.pipeline import candidate_hash, rank_candidates
from rag_benchmark.techniques.bge_m3.store import (
    BgeM3Store,
    ScoredCandidate,
    _multi_vector_score,
    _sparse_dot,
)
from rag_benchmark.techniques.rankify.config import RANKIFY_PRESETS


def config(**values):
    return BgeM3Config(**values)


def test_config_validation_and_index_fingerprint(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    parsed = BgeM3Config.from_dict({
        "enabled": True,
        "model": "./snapshot",
        "revision": "commit-a",
    }, tmp_path)
    assert parsed.model == str(snapshot.resolve())
    original = parsed.index_fingerprint("corpus", ChunkingConfig())
    assert original == replace(parsed, dense_weight=.8).index_fingerprint("corpus", ChunkingConfig())
    assert original != replace(parsed, revision="commit-b").index_fingerprint("corpus", ChunkingConfig())
    assert original != replace(parsed, passage_max_length=1024).index_fingerprint("corpus", ChunkingConfig())
    assert BgeM3Config.from_dict({}, tmp_path) is None
    assert BgeM3Config.from_dict({"enabled": True}, tmp_path).model == "BAAI/bge-m3"
    with pytest.raises(ValueError, match="weights"):
        BgeM3Config.from_dict({
            "enabled": True, "model": "BAAI/bge-m3",
            "dense_weight": 0, "sparse_weight": 0, "multi_vector_weight": 0,
        }, tmp_path)


def test_sparse_and_multi_vector_scoring_are_model_native():
    left = _sparse_vector({"7": .5, "2": .25})
    right = _sparse_vector({"7": .4, "3": 9})
    assert left.indices == [2, 7]
    assert _sparse_dot(left, right) == pytest.approx(.2)
    query = np.asarray([[1., 0.], [0., 1.]])
    passage = np.asarray([[1., 0.], [.6, .8]])
    assert _multi_vector_score(query, passage) == pytest.approx(.9)


def test_embedding_wrapper_uses_raw_query_and_all_three_outputs(monkeypatch):
    observed = {}

    class Model:
        def __init__(self, path, **kwargs):
            observed["load"] = (path, kwargs)
            self.model = SimpleNamespace(
                model=SimpleNamespace(config=SimpleNamespace(hidden_size=3)),
                colbert_linear=SimpleNamespace(out_features=2),
            )

        def encode(self, texts, **kwargs):
            observed["encode"] = (texts, kwargs)
            return {
                "dense_vecs": np.asarray([[1., 0., 0.]]),
                "lexical_weights": [{"7": .5}],
                "colbert_vecs": [np.asarray([[1., 0.]])],
            }

    monkeypatch.setitem(sys.modules, "FlagEmbedding", SimpleNamespace(BGEM3FlagModel=Model))
    monkeypatch.setattr(embedding_module, "_resolved_model", lambda *args: ("/snapshot", "commit"))
    monkeypatch.setattr(embedding_module, "synchronize_cuda", lambda *args: None)
    embedder = embedding_module.BgeM3Embedding(config(), "cpu")
    encoded = embedder.encode_query("raw Indonesian query")
    assert embedder.revision == "commit"
    assert embedder.vector_size == 3
    assert embedder.multi_vector_size == 2
    assert observed["load"] == ("/snapshot", {
        "use_fp16": False, "devices": ["cpu"], "trust_remote_code": False,
    })
    assert observed["encode"][0] == ["raw Indonesian query"]
    assert observed["encode"][1]["return_dense"] is True
    assert observed["encode"][1]["return_sparse"] is True
    assert observed["encode"][1]["return_colbert_vecs"] is True
    assert encoded.sparse.indices == [7]


def _probe_embedder(error, attempts, torch_module):
    def no_grad_wrapper(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapped

    class Model:
        @no_grad_wrapper
        def encode_single_device(self, texts, batch_size, **kwargs):
            while True:
                attempts.append(int(batch_size))
                try:
                    raise error
                except RuntimeError:
                    batch_size = batch_size * 3 // 4
                except torch_module.cuda.OutOfMemoryError:
                    batch_size = batch_size * 3 // 4

        def encode(self, texts, batch_size, **kwargs):
            return self.encode_single_device(texts, batch_size, **kwargs)

    embedder = object.__new__(embedding_module.BgeM3Embedding)
    embedder.config = config(embedding_batch_size=4)
    embedder.device = "cuda"
    embedder.model = Model()
    return embedder


def test_flagembedding_probe_guard_propagates_non_oom_runtime_error_immediately(monkeypatch):
    class CudaOutOfMemoryError(RuntimeError):
        pass

    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(OutOfMemoryError=CudaOutOfMemoryError),
        no_grad=lambda: nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    attempts = []
    original = RuntimeError("unsupported CUDA kernel")
    embedder = _probe_embedder(original, attempts, torch_module)

    with pytest.raises(RuntimeError, match="unsupported CUDA kernel") as raised:
        embedder.encode_query("question")

    assert raised.value is original
    assert attempts == [4]
    assert "encode_single_device" not in embedder.model.__dict__


def test_flagembedding_probe_guard_stops_oom_at_batch_one_with_original_traceback(monkeypatch):
    class CudaOutOfMemoryError(RuntimeError):
        pass

    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(OutOfMemoryError=CudaOutOfMemoryError),
        no_grad=lambda: nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    attempts = []
    original = CudaOutOfMemoryError("CUDA out of memory")
    embedder = _probe_embedder(original, attempts, torch_module)

    with pytest.raises(CudaOutOfMemoryError, match="CUDA out of memory") as raised:
        embedder.encode_query("question")

    assert raised.value is original
    assert attempts == [4, 3, 2, 1]
    assert "encode_single_device" not in embedder.model.__dict__
    frames = []
    traceback = raised.value.__traceback__
    while traceback is not None:
        frames.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert "encode_single_device" in frames


def test_retrieval_propagates_runtime_error_with_traceback(tmp_path, monkeypatch):
    class Embedder:
        revision = "resolved-commit"

        def __init__(self, *args):
            pass

        def encode_query(self, text):
            raise RuntimeError("visible inference failure")

    class Store:
        def __init__(self, *args):
            pass

        def close(self):
            pass

    monkeypatch.setattr(runner, "BgeM3Embedding", Embedder)
    monkeypatch.setattr(runner, "BgeM3Store", Store)
    monkeypatch.setattr(runner, "_release_gpu", lambda: None)
    spec = SimpleNamespace(bge_m3=config(), dataset="its", candidates=1)
    question = QARecord("q1", "its", "id", "question", ["answer"], "test")

    with pytest.raises(RuntimeError, match="visible inference failure"):
        runner._retrieve(
            spec,
            [question],
            tmp_path / "index",
            {"model_revision": "resolved-commit", "index_fingerprint": "hash"},
            tmp_path / "candidates.jsonl",
            "cuda",
        )


def test_five_variants_share_pool_but_produce_separate_rankings():
    items = [
        ScoredCandidate(RetrievedEvidence("a", "a", "a", 0), .9, .1, .2),
        ScoredCandidate(RetrievedEvidence("b", "b", "b", 0), .1, .8, .3),
        ScoredCandidate(RetrievedEvidence("c", "c", "c", 0), .2, .2, .9),
    ]
    pool_hash = candidate_hash(items)
    rankings = {
        variant: [item.chunk_id for item in rank_candidates(items, variant, (.4, .2, .4))]
        for variant in VARIANTS
    }
    assert rankings["bge_m3_dense"][0] == "a"
    assert rankings["bge_m3_sparse"][0] == "b"
    assert rankings["bge_m3_multi_vector"][0] == "c"
    assert rankings["bge_m3_all"][0] == "c"
    assert all(candidate_hash(items) == pool_hash for _ in rankings)


def _chunk(identifier: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        identifier, identifier, "its", "id", text, text, text, identifier,
        f"{identifier}.txt", identifier,
    )


def test_store_has_one_collection_with_three_vector_modes(tmp_path):
    client = QdrantClient(location=":memory:")
    store = BgeM3Store(tmp_path, "test", client=client)
    store.recreate("id", 2, 2)
    encoded = BgeM3Batch(
        [np.asarray([1., 0.]), np.asarray([0., 1.])],
        [_sparse_vector({1: 1}), _sparse_vector({2: 1})],
        [np.asarray([[1., 0.]]), np.asarray([[0., 1.]])],
    )
    store.add("id", [_chunk("a", "alpha"), _chunk("b", "beta")], encoded)
    query = BgeM3Encoding(
        np.asarray([1., 0.]), _sparse_vector({2: 1}), np.asarray([[0., 1.]]),
    )
    candidates = store.search("id", query, 1)
    assert {item.evidence.chunk_id for item in candidates} == {"a", "b"}
    assert candidate_hash(candidates)
    collection = client.get_collection(store.collection("id")).config.params
    assert set(collection.vectors) == {"dense", "colbert"}
    assert set(collection.sparse_vectors) == {"learned_sparse"}
    assert collection.sparse_vectors["learned_sparse"].modifier is None
    client.close()


def test_store_fetches_large_multivectors_in_bounded_batches(tmp_path, monkeypatch):
    monkeypatch.setenv("BGE_M3_VECTOR_FETCH_BATCH_SIZE", "7")
    inner = QdrantClient(location=":memory:")

    class TrackingClient:
        def __init__(self, client):
            self.client = client
            self.retrieve_batch_sizes = []

        def __getattr__(self, name):
            return getattr(self.client, name)

        def retrieve(self, collection_name, ids, **kwargs):
            self.retrieve_batch_sizes.append(len(ids))
            return self.client.retrieve(collection_name, ids, **kwargs)

    client = TrackingClient(inner)
    store = BgeM3Store(tmp_path, "bounded", client=client)
    store.recreate("id", 2, 2)
    chunks = [_chunk(f"dense-{number}", "dense") for number in range(20)]
    chunks += [_chunk(f"sparse-{number}", "sparse") for number in range(20)]
    store.add(
        "id",
        chunks,
        BgeM3Batch(
            [np.asarray([1., 0.])] * 20 + [np.asarray([0., 1.])] * 20,
            [_sparse_vector({1: 1})] * 20 + [_sparse_vector({2: 1})] * 20,
            [np.asarray([[1., 0.], [0., 1.]])] * 40,
        ),
    )

    candidates = store.search(
        "id",
        BgeM3Encoding(
            np.asarray([1., 0.]), _sparse_vector({2: 1}), np.asarray([[1., 0.]]),
        ),
        20,
    )

    assert len(candidates) == 40
    assert len(client.retrieve_batch_sizes) > 1
    assert max(client.retrieve_batch_sizes) <= 7
    inner.close()


@pytest.fixture
def bge_environment(tmp_path, monkeypatch):
    client = QdrantClient(location=":memory:")
    events, prompts = [], []

    class Embedder:
        vector_size = 2
        multi_vector_size = 2
        revision = "resolved-commit"

        def __init__(self, *args):
            events.append("load-bge-m3")

        @staticmethod
        def _one(text):
            alpha = 1.0 if "alpha" in text.casefold() else 0.0
            dense = np.asarray([alpha, 1.0 - alpha])
            sparse = _sparse_vector({1 if alpha else 2: 1})
            return dense, sparse, dense.reshape(1, 2)

        def encode_passages(self, texts):
            values = [self._one(text) for text in texts]
            return BgeM3Batch(
                [item[0] for item in values], [item[1] for item in values],
                [item[2] for item in values],
            )

        def encode_query(self, text):
            dense, sparse, multi = self._one(text)
            return BgeM3Encoding(dense, sparse, multi)

        def __del__(self):
            events.append("unload-bge-m3")

    class Store(BgeM3Store):
        def __init__(self, root, namespace):
            super().__init__(root, namespace, client=client)

    class Runtime:
        def __init__(self):
            events.append("load-generator")

        def generate_prompt(self, prompt):
            prompts.append(prompt)
            return "answer one two three four"

    monkeypatch.setattr(runner, "BgeM3Embedding", Embedder)
    monkeypatch.setattr(runner, "BgeM3Store", Store)
    monkeypatch.setattr(runner, "resolve_torch_device", lambda *args: "cpu")
    monkeypatch.setattr(runner, "_release_gpu", gc.collect)
    monkeypatch.setattr(GenerationSpec, "runtime", lambda self: Runtime())
    yield tmp_path, events, prompts
    client.close()


def spec_and_bundle(tmp_path):
    documents = [
        DocumentRecord("alpha", "its", "id", "Alpha", "alpha " * 80, "alpha.txt"),
        DocumentRecord("beta", "its", "id", "Beta", "beta " * 80, "beta.txt"),
    ]
    questions = [
        QARecord("q1", "its", "id", "alpha question", ["answer one two three four"],
                 "test", source_ids=["alpha"]),
    ]
    spec = ExperimentSpec(
        "its", tmp_path, tmp_path / "indexes", Path("python"), "dense", "reranker",
        1, 1, 60, "all", RANKIFY_PRESETS,
        GenerationSpec("huggingface", "local/generator", 0, 64),
        None, config(revision="resolved-commit"),
    )
    return spec, DatasetBundle(documents, questions)


def test_runner_writes_five_isolated_outputs_and_dense_ablation(bge_environment):
    tmp_path, events, prompts = bge_environment
    spec, bundle = spec_and_bundle(tmp_path)
    manifest = runner.run_index(spec, bundle)
    reports = runner.run_evaluate(spec, bundle, tmp_path / "run")
    assert manifest["model_revision"] == "resolved-commit"
    assert set(reports) == set(VARIANTS)
    hashes = []
    for variant in VARIANTS:
        rows = read_predictions(tmp_path / "run" / variant / "predictions.jsonl")
        assert len(rows) == 1
        assert rows[0]["technique"] == variant
        assert rows[0]["diagnostics"]["model_revision"] == "resolved-commit"
        hashes.append(rows[0]["diagnostics"]["candidate_pool_hash"])
    assert len(set(hashes)) == 1
    ablation = json.loads((tmp_path / "run" / "bge-m3-ablation.json").read_text())
    assert ablation["baseline"] == "bge_m3_dense"
    assert ablation["candidate_pools_identical"] is True
    assert set(ablation["variant_minus_dense"]) == set(VARIANTS[1:])
    benchmark._write_leaderboard(tmp_path / "run", reports)
    leaderboard = json.loads((tmp_path / "run" / "leaderboard.json").read_text())
    assert {row["technique"] for row in leaderboard} == set(VARIANTS)
    assert events.index("unload-bge-m3") < events.index("load-generator")
    assert all("answer one two three four" not in prompt for prompt in prompts)


@pytest.mark.parametrize("operation", ["index", "evaluate"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_each_variant_routes_through_cli(tmp_path, monkeypatch, capsys, operation, variant):
    from rag_benchmark import cli

    spec, bundle = spec_and_bundle(tmp_path)
    monkeypatch.setattr(ExperimentSpec, "load", lambda path: spec)
    monkeypatch.setattr("rag_benchmark.shared.datasets.load_dataset", lambda *args: bundle)
    calls = []
    monkeypatch.setattr(runner, "run_index", lambda loaded, data: calls.append(("index", loaded, data)) or {"ok": True})
    monkeypatch.setattr(
        runner, "run_evaluate",
        lambda loaded, data, output, variants: calls.append(("evaluate", loaded, data, variants)) or {"ok": True},
    )
    arguments = [operation, "--technique", variant, "--dataset", "its", "--config", "experiment.yaml"]
    arguments += ["--source", str(tmp_path)] if operation == "index" else ["--output", str(tmp_path / "run")]
    cli.main(arguments)
    assert calls[0][0] == operation
    if operation == "evaluate":
        assert calls[0][3] == (variant,)
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_full_benchmark_builds_bge_index_once(tmp_path, monkeypatch):
    spec, bundle = spec_and_bundle(tmp_path)
    calls = []

    class Worker:
        def __init__(self, *args):
            pass

        def invoke(self, request):
            return {"status": "ok"}

    monkeypatch.setattr(ExperimentSpec, "load", lambda path: spec)
    monkeypatch.setattr(benchmark, "load_dataset", lambda *args: bundle)
    monkeypatch.setattr(benchmark, "RankifyWorkerClient", Worker)
    monkeypatch.setattr(benchmark, "run_index", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(benchmark, "_index_rankify", lambda *args, **kwargs: {"status": "ok"})
    monkeypatch.setattr(
        benchmark, "index_bge_m3",
        lambda loaded, data: calls.append((loaded, data)) or {"status": "complete"},
    )
    result = benchmark.run_benchmark(tmp_path / "config.yaml", "index", tmp_path / "run")
    assert len(calls) == 1
    assert calls[0] == (spec, bundle)
    assert result["result"]["index"]["bge_m3"] == {"status": "complete"}
