from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rag_benchmark.shared.records import DocumentRecord
from rag_benchmark.shared.comparison import run_compare_techniques
from rag_benchmark.shared.metrics import write_jsonl
from rag_benchmark.techniques.piperag.config import PipeRagConfig
from rag_benchmark.techniques.piperag.performance import PerformanceProfile
from rag_benchmark.techniques.piperag.schedule import StaleAsyncSchedule
from rag_benchmark.techniques.retro.client import InferenceServiceClient, JsonGrpcClient, ServiceError
from rag_benchmark.techniques.retro.config import PaperEvaluationConfig, RetroConfig, ServiceConfig
from rag_benchmark.techniques.retro.indexing import choose_nlist, clamp_nprobe, deterministic_holdout
from rag_benchmark.techniques.retro.onnx_runtime import (
    clone_past_key_values,
    parity_report,
    service_retrieval_endpoint,
    uses_stale_schedule,
)
from rag_benchmark.techniques.retro.prompt import build_qa_prompt_ids, extract_answer
from rag_benchmark.techniques.retro.retrieval_service import RetrievalService
from rag_benchmark.techniques.retro.grpc_service import create_json_server
from rag_benchmark.techniques.retro.runner import run_index


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) % 251 + 1 for char in text]

    def decode(self, values, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(value) for value in values if value)


def _document(identifier: str, language: str) -> DocumentRecord:
    return DocumentRecord(identifier, "x", language, identifier, f"text {identifier}", f"{identifier}.txt")


def test_prompt_is_truncated_and_left_padded_to_exactly_64_tokens():
    tokenizer = Tokenizer()
    ids = build_qa_prompt_ids(tokenizer, "q" * 200, 64)
    assert len(ids) == 64
    assert ids[-len(tokenizer.encode("\nAnswer: ")):] == tokenizer.encode("\nAnswer: ")
    short = build_qa_prompt_ids(tokenizer, "short", 64)
    assert len(short) == 64 and short[0] == 0
    assert extract_answer("Question: x\nAnswer: result") == "result"


def test_inference_uses_compose_internal_retrieval_endpoint(monkeypatch):
    request = {"retrieval_endpoint": "127.0.0.1:50051"}
    assert service_retrieval_endpoint(request) == "127.0.0.1:50051"
    monkeypatch.setenv("RETRO_RETRIEVAL_ENDPOINT", "retrieval:50051")
    assert service_retrieval_endpoint(request) == "retrieval:50051"


def test_parity_check_isolates_mutable_past_key_values():
    original = [np.ones((1, 2), dtype=np.float32)]
    isolated = clone_past_key_values(original)
    isolated[0][0, 0] = 99
    isolated[0] = np.zeros((1, 3), dtype=np.float32)
    assert original[0].shape == (1, 2)
    assert original[0][0, 0] == 1


@pytest.mark.parametrize("chunks,expected", [(1, 1), (3, 1), (16, 4), (100, 8), (2_000_000, 1024)])
def test_nlist_is_bounded_power_of_two_and_nprobe_clamped(chunks, expected):
    assert choose_nlist(chunks) == expected
    assert clamp_nprobe(0, expected) == 1
    assert clamp_nprobe(9999, expected) == expected


def test_holdout_is_deterministic_per_language_and_never_leaks():
    documents = [_document(f"en-{index}", "en") for index in range(11)]
    documents += [_document(f"it-{index}", "it") for index in range(3)]
    train_a, holdout_a = deterministic_holdout(documents, .10, 42)
    train_b, holdout_b = deterministic_holdout(list(reversed(documents)), .10, 42)
    assert [item.id for item in holdout_a] == [item.id for item in holdout_b]
    assert {item.id for item in train_a}.isdisjoint(item.id for item in holdout_a)
    assert {item.language for item in holdout_a} == {"en", "it"}


def test_run_index_creates_host_manifest_directory_before_service_build(tmp_path):
    index_root = tmp_path / "indexes"
    spec = SimpleNamespace(
        dataset="its",
        index_root=index_root,
        services=SimpleNamespace(retrieval_index_root="/data/indexes"),
        retro=SimpleNamespace(
            tokenizer="t5-base",
            retriever_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=64,
            fingerprint=lambda: "retro-fingerprint",
        ),
        paper=SimpleNamespace(enabled=False, holdout_fraction=.10, seed=42),
    )
    bundle = SimpleNamespace(documents=[_document("id-1", "id")])

    class Client:
        @staticmethod
        def build_index(request):
            assert (index_root / "retro" / "its").is_dir()
            return {"index": {"chunks": len(request["documents"])}}

    manifest = run_index(spec, bundle, Client())

    assert manifest["collections"]["its"]["chunks"] == 1
    assert (index_root / "retro" / "its" / "index-manifest.json").is_file()


def test_chunk_records_keep_source_and_continuation_boundaries():
    class TinyTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens=True):
            del add_special_tokens
            return [int(value) for value in text.split()]

    chunks, records = RetrievalService._chunk_documents([
        {"id": "a", "text": "1 2 3 4 5", "source_path": "a.txt"},
        {"id": "b", "text": "6 7", "source_path": "b.txt"},
    ], TinyTokenizer(), 4)
    assert chunks.tolist() == [[1, 2, 3, 4], [5, 0, 0, 0], [6, 7, 0, 0]]
    assert [(row["source_id"], row["ordinal"]) for row in records] == [("a", 0), ("a", 1), ("b", 0)]


def test_pipe_schedule_reuses_previous_interval_then_consumes_pending():
    release = threading.Event()

    def retrieve(query, seq_len):
        if query == "second":
            release.wait(1)
        return f"{query}:{seq_len}"

    with StaleAsyncSchedule(retrieve) as schedule:
        assert schedule.boundary("first", 64) == "first:64"
        assert schedule.boundary("second", 128) == "first:64"
        release.set()
        assert schedule.boundary("third", 192) == "second:128"
        assert schedule.calls == 3


def test_dynamic_performance_model_selects_largest_probe_within_budget():
    profile = PerformanceProfile(
        decoder_latency_ms=(2.0,) * 256,
        encoder_latency_ms=3.0,
        embedding_latency_ms=4.0,
        search_latency_ms_per_nprobe=5.0,
        extra_overhead_ms=4.0,
    )
    assert profile.choose_nprobe(64, 16, nlist=64, maximum=64) == 4
    assert profile.choose_nprobe(64, 64, nlist=8, maximum=64) == 8


def test_piperag_without_staleness_uses_fresh_path_and_parity_is_enforced():
    assert not uses_stale_schedule("retro", 0)
    assert not uses_stale_schedule("piperag", 0)
    assert uses_stale_schedule("piperag", 64)
    expected_encoder = [np.ones((1, 2)), np.zeros((1, 2))]
    expected_logits = np.array([[[0.1, 0.9, 0.2]]])
    report = parity_report(
        expected_encoder, [value + 1e-5 for value in expected_encoder],
        expected_logits, expected_logits + 1e-5, 1e-3,
    )
    assert report["token_equal"] is True
    assert report["encoder_atol"] == report["decoder_atol"] == 1e-3
    with pytest.raises(RuntimeError, match="parity failed"):
        parity_report(expected_encoder, expected_encoder, expected_logits, expected_logits[..., ::-1], 1e-3)

    encoder_drift = [value + 1.5e-3 for value in expected_encoder]
    relaxed = parity_report(
        expected_encoder, encoder_drift, expected_logits, expected_logits + 1.5e-3,
        encoder_tolerance=2e-3, decoder_tolerance=2e-3,
    )
    assert relaxed["encoder_max_abs"] < relaxed["encoder_atol"]
    with pytest.raises(RuntimeError, match=r"encoder_max_abs=.*atol=0.001"):
        parity_report(
            expected_encoder, encoder_drift, expected_logits, expected_logits,
            encoder_tolerance=1e-3, decoder_tolerance=2e-3,
        )


def test_typed_configuration_defaults_and_validation(tmp_path):
    retro = RetroConfig.from_dict({
        "checkpoint": "model.ckpt", "model_config": "retro.json",
        "onnx_encoder": "encoder.onnx", "onnx_decoder": "decoder.onnx",
    }, tmp_path)
    assert retro.checkpoint == (tmp_path / "model.ckpt").resolve()
    assert retro.chunk_size == retro.retrieval_interval == 64
    assert ServiceConfig.from_dict({}).timeout_seconds == 120
    assert PipeRagConfig.from_dict({}).stale_steps == 64
    assert PaperEvaluationConfig.from_dict({}).intervals == (8, 16, 32, 64)
    with pytest.raises(ValueError, match="chunk_size=64"):
        RetroConfig.from_dict({
            "checkpoint": "a", "model_config": "b", "onnx_encoder": "c", "onnx_decoder": "d",
            "chunk_size": 32,
        }, tmp_path)


def test_json_grpc_contract_retries_once_and_forwards_deadline():
    calls = []

    def transport(method, payload, timeout):
        calls.append((method, payload, timeout))
        if len(calls) == 1:
            raise TimeoutError("deadline")
        return {"protocol_version": 1, "status": "ok", "value": 7}

    client = JsonGrpcClient("fake", "rag.Test", ServiceConfig(timeout_seconds=9), transport)
    assert client.call("Generate", {"index_id": "x"})["value"] == 7
    assert len(calls) == 2 and calls[0][2] == 9
    assert calls[0][1]["protocol_version"] == 1


def test_json_grpc_contract_rejects_protocol_and_index_mismatch():
    config = ServiceConfig()
    wrong_protocol = JsonGrpcClient(
        "fake", "rag.Test", config,
        lambda *_: {"protocol_version": 2, "status": "ok"},
    )
    with pytest.raises(ServiceError, match="protocol version mismatch"):
        wrong_protocol.call("Health", {}, retry=False)
    wrong_index = JsonGrpcClient(
        "fake", "rag.Test", config,
        lambda *_: {"protocol_version": 1, "status": "error", "error": "index fingerprint mismatch"},
    )
    with pytest.raises(ServiceError, match="index fingerprint mismatch"):
        wrong_index.call("LoadIndex", {}, retry=False)


def test_compare_allows_cross_model_runs_but_labels_controlled_pair(tmp_path):
    def row(technique, model):
        return {
            "id": "qa-1", "dataset": "its", "language": "id", "split": "test",
            "technique": technique, "model_fingerprint": model, "corpus_fingerprint": "corpus",
            "prediction": "answer", "references": ["answer"], "latency_ms": 1,
            "status": "ok", "available": True, "diagnostics": {},
        }

    baseline, candidate = tmp_path / "dillema.jsonl", tmp_path / "retro.jsonl"
    write_jsonl(baseline, [row("dillema", "model-a")])
    write_jsonl(candidate, [row("retro", "model-b")])
    compared = run_compare_techniques(baseline, candidate, tmp_path / "cross")
    assert compared["comparability"] == "cross_model"
    write_jsonl(baseline, [row("retro", "same-checkpoint")])
    write_jsonl(candidate, [row("piperag", "same-checkpoint")])
    controlled = run_compare_techniques(baseline, candidate, tmp_path / "controlled")
    assert controlled["comparability"] == "controlled_same_checkpoint"
    assert controlled["warning"] is None


def test_real_grpc_contract_with_fake_inference_service_timeout_retry_and_profile():
    calls = {"generate": 0}

    def health(request):
        return {"protocol_version": request["protocol_version"], "status": "ok", "role": "fake"}

    def generate(request):
        calls["generate"] += 1
        time.sleep(.3)
        return {"protocol_version": request["protocol_version"], "status": "ok", "prediction": "x"}

    def profile(request):
        return {
            "protocol_version": request["protocol_version"], "status": "ok",
            "profile": {"hardware_fingerprint": "stale-hardware", "cache_reason": "rebuild_required"},
        }

    server, port = create_json_server(
        "rag.InferenceService", {"Health": health, "Generate": generate, "ProfileGeneration": profile},
        "127.0.0.1", 0,
    )
    server.start()
    client = InferenceServiceClient(ServiceConfig(
        inference_endpoint=f"127.0.0.1:{port}", timeout_seconds=.1,
    ))
    try:
        assert client.health()["role"] == "fake"
        response = client.profile_generation({})
        assert response["profile"]["hardware_fingerprint"] == "stale-hardware"
        with pytest.raises(ServiceError, match="after 2 attempt"):
            client.generate({"question": "timeout"})
        assert calls["generate"] == 2
    finally:
        client.close()
        server.stop(0).wait()
