from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, replace

from ...shared.records import RetrievedEvidence, TechniqueResult
from .config import VARIANTS
from .prompt import base_prompt, evidence_prompt, no_retrieval_prompt
from .reflection import passage_score, postprocess, retrieval_logprobs, retrieval_score


@dataclass(slots=True)
class SelfRagOutcome:
    prediction: str
    raw_prediction: str
    evidence: list[RetrievedEvidence]
    context: str
    retrieval_used: bool
    generation_ms: float
    diagnostics: dict = field(default_factory=dict)


def _elapsed(started: float) -> float:
    return (time.perf_counter() - started) * 1000


class SelfRagPipeline:
    def __init__(self, llm, config):
        self.llm = llm
        self.config = config

    def run(self, question: str, passages: list[RetrievedEvidence], variants=VARIANTS) -> dict[str, SelfRagOutcome]:
        if not variants or any(name not in VARIANTS for name in variants):
            raise ValueError("Unsupported Self-RAG variant")
        config, ids = self.config, self.llm.ids
        prompt = base_prompt(question, config.instruction)
        passages = passages[:config.ndocs]
        timings: dict[str, float] = {}
        decision: dict = {}
        if "self_rag_adaptive" in variants:
            started = time.perf_counter()
            probe = self.llm.generate([prompt], max_new_tokens=1)[0]
            timings["retrieval_probe"] = _elapsed(started)
            logprobs = retrieval_logprobs(probe, ids)
            score = retrieval_score(logprobs)
            decision = {
                "retrieval_logprobs": logprobs,
                "retrieval_score": None if math.isnan(score) else score,
                "retrieval_threshold": config.threshold,
                "retrieve": bool(score > config.threshold),
            }
        ranked = []
        if passages and ("self_rag_always" in variants or decision.get("retrieve")):
            started = time.perf_counter()
            generations = self.llm.generate([
                evidence_prompt(prompt, str(item.metadata.get("title") or ""), item.text)
                for item in passages
            ])
            timings["passage_generation"] = _elapsed(started)
            scored = [(item, generation, passage_score(generation, ids, config))
                      for item, generation in zip(passages, generations)]
            ranked = sorted(scored, key=lambda value: value[2].final, reverse=True)
        outcomes = {}
        if "self_rag_always" in variants:
            outcomes["self_rag_always"] = self._from_passages(
                ranked, timings.get("passage_generation", 0.0), timings, {},
            )
        if "self_rag_adaptive" in variants:
            if decision["retrieve"] and ranked:
                outcomes["self_rag_adaptive"] = self._from_passages(
                    ranked, timings["retrieval_probe"] + timings["passage_generation"], timings, decision,
                )
            else:
                started = time.perf_counter()
                generation = self.llm.generate([no_retrieval_prompt(prompt)])[0]
                timings["no_retrieval_generation"] = _elapsed(started)
                outcomes["self_rag_adaptive"] = SelfRagOutcome(
                    postprocess(generation.text), generation.text, [], "", False,
                    timings["retrieval_probe"] + timings["no_retrieval_generation"],
                    {**decision, "stage_latency_ms": dict(timings)},
                )
        return outcomes

    @staticmethod
    def _from_passages(ranked, generation_ms: float, timings: dict, decision: dict) -> SelfRagOutcome:
        if not ranked:
            return SelfRagOutcome("", "", [], "", False, generation_ms,
                                  {**decision, "stage_latency_ms": dict(timings)})
        evidence = [replace(item, rerank_score=score.final) for item, _, score in ranked]
        best_item, best_generation, _ = ranked[0]
        return SelfRagOutcome(
            postprocess(best_generation.text), best_generation.text, evidence, best_item.text, True,
            generation_ms,
            {
                **decision,
                "selected_chunk_id": best_item.chunk_id,
                "passage_scores": [
                    {"chunk_id": item.chunk_id, "raw_prediction": generation.text, **asdict(score)}
                    for item, generation, score in ranked
                ],
                "stage_latency_ms": dict(timings),
            },
        )


class PreparedAnswer:
    def __init__(self, outcome: SelfRagOutcome | None, error: str | None = None):
        self.outcome = outcome
        self.error = error

    def answer(self, collection: str, question: str) -> TechniqueResult:
        if self.error or self.outcome is None:
            raise RuntimeError(self.error or "Self-RAG outcome is missing")
        outcome = self.outcome
        return TechniqueResult(outcome.prediction, outcome.context, outcome.evidence, {
            **outcome.diagnostics,
            "retrieval_used": outcome.retrieval_used,
            "raw_prediction": outcome.raw_prediction,
            "generation_latency_ms": outcome.generation_ms,
        })
