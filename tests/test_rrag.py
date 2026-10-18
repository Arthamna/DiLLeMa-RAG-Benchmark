import numpy as np
import pytest
from qdrant_client import QdrantClient

from rag_benchmark.models import DocumentRecord
from rag_benchmark.cli import run_compare_techniques, run_train
from rag_benchmark.metrics import write_jsonl
from rag_benchmark.shared.config import PipelineConfig
from rag_benchmark.store import QdrantStore
from rag_benchmark.techniques import get_technique
from rag_benchmark.techniques.rrag.config import RRAGConfig
from rag_benchmark.techniques.rrag.features import compute_retrieval_features
from rag_benchmark.techniques.rrag.pipeline import RRAGTechnique


class FakeEmbedder:
    vector_size = 2

    def encode_query(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)

    def encode_passages(self, texts):
        return np.asarray([
            [1.0, 0.0] if "alpha" in text else [0.0, 1.0]
            for text in texts
        ], dtype=np.float32)


def test_registry_and_rrag_feature_values():
    assert not get_technique("dillema").supports_training
    assert get_technique("rrag").supports_training
    query = np.array([1.0, 0.0])
    passages = np.array([[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]])
    features, order = compute_retrieval_features(query, passages, top_k=3)
    assert order.tolist() == [1, 2, 0]
    assert features.shape == (3, 3)
    assert features[0, 0] == pytest.approx(1.0)
    assert features[0, 2] == pytest.approx(1.0)


def test_dillema_training_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not train"):
        run_train("dillema", "its", None, tmp_path, tmp_path / "model")


def test_technique_comparison_requires_matching_run_identity(tmp_path):
    common = {
        "id": "qa-1", "dataset": "its", "language": "id", "split": "test",
        "model_fingerprint": "same", "references": ["one two three four"],
        "prediction": "one two three four", "status": "ok", "available": True,
        "diagnostics": {}, "latency_ms": 1.0,
    }
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(baseline, [{**common, "technique": "dillema"}])
    write_jsonl(candidate, [{**common, "technique": "rrag"}])
    result = run_compare_techniques(baseline, candidate, tmp_path / "comparison")
    assert result["candidate_minus_baseline"]["id"]["BLEU_raw"] == 0.0
    write_jsonl(candidate, [{**common, "id": "different", "technique": "rrag"}])
    with pytest.raises(ValueError, match="identical"):
        run_compare_techniques(baseline, candidate, tmp_path / "bad")


def test_rrag_hybrid_candidates_are_reordered_and_labeled(tmp_path):
    store = QdrantStore(tmp_path, client=QdrantClient(location=":memory:"))
    pipeline = RRAGTechnique(
        store, FakeEmbedder(), pipeline_config=PipelineConfig(min_chunk_chars=1),
        rrag_config=RRAGConfig(top_k=2),
    )
    documents = [
        DocumentRecord("alpha", "its", "id", "Alpha", "alpha evidence enough words", "alpha.txt", {"categories": ["a"]}),
        DocumentRecord("beta", "its", "id", "Beta", "beta evidence enough words", "beta.txt", {"categories": ["b"]}),
    ]
    pipeline.index("its", documents)
    context = pipeline.retrieve("its", "alpha question")
    assert context.evidence[0].source_id == "alpha"
    assert context.features.shape == (2, 3)
    assert pipeline.labels_for(context, ["alpha"]) == [1.0, 0.0]


def test_r2former_shape_and_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from rag_benchmark.techniques.rrag.r2former import R2AGAdapter

    config = RRAGConfig(top_k=3, d_model=8, n_head=2)
    adapter = R2AGAdapter(config, llm_hidden_size=12)
    features = torch.zeros((1, 3, 3))
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    projected, logits, loss = adapter(features, labels)
    assert projected.shape == (1, 3, 12)
    assert logits.shape == (1, 3)
    assert loss.ndim == 0
    checkpoint = adapter.save_checkpoint(tmp_path, "fingerprint", "id")
    loaded, payload = R2AGAdapter.load_checkpoint(checkpoint, "fingerprint")
    assert loaded.llm_hidden_size == 12
    assert payload["language"] == "id"


def test_rrag_retrieval_to_generation_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    from rag_benchmark.techniques.rrag.r2former import R2AGAdapter

    class FakeRuntime:
        device = torch.device("cpu")

        def generate_with_injections(self, prompt, retrieval_embeddings):
            assert prompt.count("<R>") == retrieval_embeddings.shape[1]
            return "jawaban benar empat kata"

    store = QdrantStore(tmp_path, client=QdrantClient(location=":memory:"))
    config = RRAGConfig(top_k=2, d_model=8, n_head=2)
    pipeline = RRAGTechnique(
        store, FakeEmbedder(), runtime=FakeRuntime(), adapter=R2AGAdapter(config, 12),
        pipeline_config=PipelineConfig(min_chunk_chars=1), rrag_config=config,
    )
    pipeline.index("its", [
        DocumentRecord("alpha", "its", "id", "Alpha", "alpha evidence enough words", "alpha.txt"),
        DocumentRecord("beta", "its", "id", "Beta", "beta evidence enough words", "beta.txt"),
    ])
    result = pipeline.answer("its", "alpha question")
    assert result.prediction == "jawaban benar empat kata"
    assert len(result.diagnostics["retrieval_features"]) == 2

