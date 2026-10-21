import logging
import sys
from types import SimpleNamespace

from rag_benchmark.shared.device import resolve_torch_device


def _fake_torch(available: bool):
    return SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: available,
        get_device_name=lambda _index: "Test GPU",
    ))


def test_device_falls_back_to_cpu_with_log(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False))
    with caplog.at_level(logging.WARNING):
        assert resolve_torch_device("RAG evaluation", "auto") == "cpu"
    assert "tetap dilanjutkan menggunakan CPU" in caplog.text


def test_device_uses_detected_cuda(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    with caplog.at_level(logging.INFO):
        assert resolve_torch_device("BGE reranker", "auto") == "cuda"
    assert "Test GPU" in caplog.text


def test_reranker_receives_cuda_and_batch_size(monkeypatch):
    captured = {}

    class FakeCrossEncoder:
        def __init__(self, model_name, device):
            captured.update(model_name=model_name, device=device)

        def predict(self, pairs, **kwargs):
            captured.update(pairs=pairs, **kwargs)
            return [0.75] * len(pairs)

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    monkeypatch.setenv("RERANK_BATCH_SIZE", "8")

    from rag_benchmark.techniques.dillema.rerank import CrossEncoderReranker

    reranker = CrossEncoderReranker("test/bge", device="cuda")
    assert reranker.score("q", ["a", "b"]) == [0.75, 0.75]
    assert captured["device"] == "cuda"
    assert captured["batch_size"] == 8
    assert captured["show_progress_bar"] is False
