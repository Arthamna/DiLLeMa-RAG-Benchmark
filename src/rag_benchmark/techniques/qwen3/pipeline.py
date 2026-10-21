from __future__ import annotations

import time
from dataclasses import replace

from ...shared.records import RetrievedEvidence, TechniqueResult
from .config import fingerprint
from .generation import base_prompt


def candidate_hash(evidence: list[RetrievedEvidence]) -> str:
    return fingerprint({"ordered_chunk_ids": [item.chunk_id for item in evidence]})


def rerank_candidates(question, evidence, reranker):
    scores = reranker.score(question, [item.text for item in evidence])
    if len(scores) != len(evidence):
        raise RuntimeError("Qwen3 reranker returned an incomplete candidate pool")
    ranked = [replace(item, rerank_score=float(score)) for item, score in zip(evidence, scores)]
    return sorted(ranked, key=lambda item: item.rerank_score, reverse=True)


class Qwen3Pipeline:
    """Generate from one prepared pool; embedding and reranker are already released."""

    def __init__(self, prepared: dict, runtime, top_k: int, reranked: bool):
        self.prepared = prepared
        self.runtime = runtime
        self.top_k = top_k
        self.reranked = reranked
        self.name = "qwen3_embedding_rerank" if reranked else "qwen3_embedding"
        self.diagnostics = {"generation_latency_ms": 0.0, "packing_latency_ms": 0.0}

    def answer(self, collection: str, question: str) -> TechniqueResult:
        error = self.prepared.get("error") or (self.prepared.get("rerank_error") if self.reranked else None)
        if error:
            raise RuntimeError(error)
        items = self.prepared["ranked" if self.reranked else "candidates"][:self.top_k]
        evidence = [RetrievedEvidence(**item) for item in items]
        started = time.perf_counter()
        context = "\n\n---\n\n".join(item.text for item in evidence)
        self.diagnostics["packing_latency_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        try:
            prediction = self.runtime.generate_prompt(base_prompt(question, context)) if context else ""
        finally:
            self.diagnostics["generation_latency_ms"] = (time.perf_counter() - started) * 1000
        return TechniqueResult(prediction, context, evidence, dict(self.diagnostics))
