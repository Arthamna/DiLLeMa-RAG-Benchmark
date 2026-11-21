from __future__ import annotations

import time
from dataclasses import replace

from ...shared.records import RetrievedEvidence, TechniqueResult
from .config import VARIANTS, fingerprint
from .generation import base_prompt
from .store import ScoredCandidate


def candidate_hash(items: list[ScoredCandidate] | list[RetrievedEvidence]) -> str:
    return fingerprint({
        "ordered_chunk_ids": [
            item.evidence.chunk_id if isinstance(item, ScoredCandidate) else item.chunk_id
            for item in items
        ],
    })


def score_for_variant(item: ScoredCandidate, variant: str, weights) -> float:
    dense_weight, sparse_weight, multi_weight = weights
    if variant == "bge_m3_dense":
        return item.dense_score
    if variant == "bge_m3_sparse":
        return item.sparse_score
    if variant == "bge_m3_multi_vector":
        return item.multi_vector_score
    if variant == "bge_m3_dense_sparse":
        return (
            dense_weight * item.dense_score + sparse_weight * item.sparse_score
        ) / (dense_weight + sparse_weight)
    if variant == "bge_m3_all":
        return (
            dense_weight * item.dense_score
            + sparse_weight * item.sparse_score
            + multi_weight * item.multi_vector_score
        ) / (dense_weight + sparse_weight + multi_weight)
    raise ValueError(f"Unsupported BGE-M3 variant: {variant}")


def rank_candidates(items: list[ScoredCandidate], variant: str, weights) -> list[RetrievedEvidence]:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported BGE-M3 variant: {variant}")
    ranked = [
        replace(item.evidence, rerank_score=score_for_variant(item, variant, weights))
        for item in items
    ]
    return sorted(ranked, key=lambda item: item.rerank_score, reverse=True)


class BgeM3Pipeline:
    """Generate from a prepared, shared BGE-M3 candidate pool."""

    def __init__(self, prepared: dict, runtime, top_k: int, variant: str, weights):
        if variant not in VARIANTS:
            raise ValueError(f"Unsupported BGE-M3 variant: {variant}")
        self.prepared = prepared
        self.runtime = runtime
        self.top_k = top_k
        self.name = variant
        self.weights = weights
        self.diagnostics = {"generation_latency_ms": 0.0, "packing_latency_ms": 0.0}

    def answer(self, collection: str, question: str) -> TechniqueResult:
        if self.prepared.get("error"):
            raise RuntimeError(self.prepared["error"])
        candidates = []
        for value in self.prepared.get("candidates", []):
            evidence = RetrievedEvidence(**value["evidence"])
            candidates.append(ScoredCandidate(
                evidence, value["dense_score"], value["sparse_score"], value["multi_vector_score"],
            ))
        evidence = rank_candidates(candidates, self.name, self.weights)[:self.top_k]
        started = time.perf_counter()
        context = "\n\n---\n\n".join(item.text for item in evidence)
        self.diagnostics["packing_latency_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        try:
            prediction = self.runtime.generate_prompt(base_prompt(question, context)) if context else ""
        finally:
            self.diagnostics["generation_latency_ms"] = (time.perf_counter() - started) * 1000
        return TechniqueResult(prediction, context, evidence, dict(self.diagnostics))
