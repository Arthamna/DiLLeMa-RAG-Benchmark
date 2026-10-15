import numpy as np
from qdrant_client import QdrantClient

from rag_benchmark.config import PipelineConfig
from rag_benchmark.models import DocumentRecord, RetrievedEvidence
from rag_benchmark.pipeline import RAGPipeline, drop_weak_evidence, is_listing, pack_parent_pages
from rag_benchmark.sparse import encode_sparse, tokenize
from rag_benchmark.store import QdrantStore


class FakeEmbedder:
    vector_size = 6
    terms = ["alpha", "beta", "gamma", "english", "italiano", "indonesia"]

    def _one(self, text):
        lowered = text.casefold()
        values = np.array([lowered.count(term) for term in self.terms], dtype=float)
        if not values.any():
            values[0] = 0.01
        return values / np.linalg.norm(values)

    def encode_passages(self, texts):
        return np.stack([self._one(text) for text in texts])

    def encode_query(self, text):
        return self._one(text)


class FakeReranker:
    def score(self, question, passages):
        key = question.casefold().split()[0]
        return [0.9 if key in passage.casefold() else 0.04 for passage in passages]


def _doc(identifier, text, language="en"):
    return DocumentRecord(identifier, "its", language, identifier, text, f"{identifier}.txt")


def test_surface_sparse_vectors_and_listing_languages():
    vector = encode_sparse("Running runners café café")
    assert len(vector.indices) == 3
    assert sorted(vector.values) == [1.0, 1.0, 2.0]
    assert tokenize("L'università café") == ["l", "università", "café"]
    assert is_listing("Apa saja mata kuliahnya?")
    assert is_listing("What are all the subjects?")
    assert is_listing("Quali sono tutti i corsi?")


def test_rerank_threshold_relative_filter_and_parent_packing():
    items = [
        RetrievedEvidence("a", "1", "one", 0.9, {"parent_id": "a", "parent_window": "parent a"}),
        RetrievedEvidence("b", "2", "two", 0.5, {"parent_id": "b", "parent_window": "parent b"}),
        RetrievedEvidence("c", "3", "three", 0.4, {"parent_id": "c", "parent_window": "parent c"}),
    ]
    kept = drop_weak_evidence(items, 0.5)
    assert [item.source_id for item in kept] == ["a", "b"]
    packed = pack_parent_pages(kept, max_parents=1)
    assert [item.source_id for item in packed.evidence] == ["a"]
    assert "parent a" in packed.text


def test_dense_sparse_rrf_with_in_memory_qdrant(tmp_path):
    client = QdrantClient(location=":memory:")
    store = QdrantStore(tmp_path, client=client)
    config = PipelineConfig(min_chunk_chars=1, rerank_min_score=0.05)
    pipeline = RAGPipeline(store, FakeEmbedder(), FakeReranker(), config)
    pipeline.index(
        "its",
        [
            _doc("alpha", "alpha english answer has enough useful words"),
            _doc("beta", "beta unrelated passage has enough useful words"),
            _doc("gamma", "gamma unrelated passage has enough useful words"),
        ],
    )
    packed = pipeline.retrieve("its", "alpha question")
    assert packed.evidence
    assert packed.evidence[0].source_id == "alpha"
    assert all(item.rerank_score >= 0.05 for item in packed.evidence)

