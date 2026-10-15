import numpy as np
from qdrant_client import QdrantClient

from rag_benchmark.cli import evaluate_questions
from rag_benchmark.config import PipelineConfig
from rag_benchmark.metrics import summarize
from rag_benchmark.models import DocumentRecord, QARecord
from rag_benchmark.pipeline import RAGPipeline
from rag_benchmark.store import QdrantStore


class SmokeEmbedder:
    vector_size = 3

    def _vector(self, text):
        lower = text.casefold()
        values = np.array([lower.count("its"), lower.count("english"), lower.count("italiano")], dtype=float)
        if not values.any():
            values[:] = 1
        return values / np.linalg.norm(values)

    def encode_passages(self, texts):
        return np.stack([self._vector(text) for text in texts])

    def encode_query(self, text):
        return self._vector(text)


class SmokeReranker:
    def score(self, question, passages):
        token = question.casefold().split()[0]
        return [0.9 if token in passage.casefold() else 0.1 for passage in passages]


class MockOpenAICompatibleGenerator:
    def generate(self, question, context):
        language = question.casefold().split()[0]
        return {
            "its": "jawaban benar terdiri dari empat kata",
            "english": "the correct answer has five words",
            "italiano": "la risposta corretta ha cinque parole",
        }[language]


def test_two_qa_per_dataset_language_from_index_to_bleu(tmp_path):
    store = QdrantStore(tmp_path, client=QdrantClient(location=":memory:"))
    config = PipelineConfig(min_chunk_chars=1, rerank_min_score=0.05)
    pipeline = RAGPipeline(store, SmokeEmbedder(), SmokeReranker(), config)
    answers = {
        "id": "jawaban benar terdiri dari empat kata",
        "en": "the correct answer has five words",
        "it": "la risposta corretta ha cinque parole",
    }
    for dataset, language, prefix in (("its", "id", "its"), ("uniqa", "en", "english"), ("uniqa", "it", "italiano")):
        collection = "its" if dataset == "its" else f"uniqa_{language}"
        documents = [
            DocumentRecord(f"{prefix}-{index}", dataset, language, f"title {index}", f"{prefix} evidence number {index} contains the answer", f"{index}.txt")
            for index in range(2)
        ]
        pipeline.index(collection, documents)
    generator = MockOpenAICompatibleGenerator()
    its_questions = [QARecord(f"i{n}", "its", "id", f"its question {n}", [answers["id"]], "all") for n in range(2)]
    uniqa_questions = [
        QARecord(f"{language}{n}", "uniqa", language, f"{prefix} question {n}", [answers[language]], "test")
        for language, prefix in (("en", "english"), ("it", "italiano"))
        for n in range(2)
    ]
    its_rows = [item.to_dict() for item in evaluate_questions(its_questions, pipeline, generator, "its")]
    uniqa_rows = [item.to_dict() for item in evaluate_questions(uniqa_questions, pipeline, generator, "uniqa")]
    assert len(its_rows) == 2 and len(uniqa_rows) == 4
    assert summarize(its_rows)["its"]["BLEU_raw"]["score"] == 100.0
    assert summarize(uniqa_rows)["strict"]["macro_average"]["BLEU_raw"] == 100.0

