from __future__ import annotations

import time
from pathlib import Path

from .config import collection_name
from .device import is_fatal_accelerator_error
from .records import PredictionRecord, QARecord, RetrievedEvidence


def _relevant_ranks(evidence: list[RetrievedEvidence], gold: list[str], limit: int = 10) -> list[int]:
    if not gold:
        return []
    normalized_gold = {str(item).casefold() for item in gold}
    ranks: list[int] = []
    matched_gold: set[str] = set()
    for rank, item in enumerate(evidence[:limit], 1):
        retrieved = {item.source_id.casefold()}
        source_path = str(item.metadata.get("source_path") or "")
        if source_path:
            retrieved.add(source_path.casefold())
            retrieved.add(Path(source_path).name.casefold())
        retrieved.update(str(alias).casefold() for alias in item.metadata.get("aliases", []))
        source_file = str(item.metadata.get("source_file") or "")
        if source_file:
            retrieved.add(source_file.casefold())
        new_matches = (normalized_gold & retrieved) - matched_gold
        if new_matches:
            ranks.append(rank)
            matched_gold.update(new_matches)
    return ranks


def evaluate_questions(
    questions: list[QARecord],
    runner,
    dataset: str,
    technique: str,
    model_fingerprint: str,
    corpus_fingerprint: str = "",
    decoding_config: dict | None = None,
) -> list[PredictionRecord]:
    predictions: list[PredictionRecord] = []
    for question in questions:
        started = time.perf_counter()
        prediction = ""
        evidence: list[RetrievedEvidence] = []
        context = ""
        diagnostics = {}
        error = None
        if not question.available:
            status = "missing_context"
        else:
            try:
                result = runner.answer(collection_name(dataset, question.language), question.question)
                prediction = result.prediction
                evidence = result.evidence
                context = result.context
                diagnostics = dict(result.diagnostics)
                if not context:
                    status = "no_evidence"
                else:
                    status = "ok" if prediction else "abstention"
            except Exception as exc:  # failures remain in the BLEU denominator
                if is_fatal_accelerator_error(exc):
                    raise RuntimeError(
                        f"Fatal CUDA error: technique={technique}, dataset={dataset}, "
                        f"question_id={question.id}, language={question.language}. "
                        f"Evaluation stopped; restart in a new process. {exc}"
                    ) from exc
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
        relevant_ranks = _relevant_ranks(evidence, question.source_ids)
        if question.source_ids:
            diagnostics["relevant_ranks"] = relevant_ranks
            diagnostics["source_recall_at_10"] = 1.0 if relevant_ranks else 0.0
            diagnostics["relevance_label"] = question.metadata.get("relevance_label")
        predictions.append(PredictionRecord(
            id=question.id,
            dataset=question.dataset,
            language=question.language,
            question=question.question,
            references=question.references,
            prediction=prediction,
            retrieved_source_ids=list(dict.fromkeys(item.source_id for item in evidence)),
            rerank_scores=[item.rerank_score for item in evidence],
            context_length=len(context),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            status=status,
            available=question.available,
            error=error,
            technique=technique,
            split=question.split,
            model_fingerprint=model_fingerprint,
            corpus_fingerprint=corpus_fingerprint,
            decoding_config=dict(decoding_config or {}),
            gold_source_ids=question.source_ids,
            diagnostics=diagnostics,
        ))
    return predictions

