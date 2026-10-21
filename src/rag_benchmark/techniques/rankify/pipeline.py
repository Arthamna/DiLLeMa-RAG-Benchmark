from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from ...shared.artifacts import _languages
from ...shared.chunking import DocumentChunker
from ...shared.config import ChunkingConfig
from ...shared.evaluation import _relevant_ranks
from ...shared.metrics import read_predictions, summarize, write_json, write_jsonl
from ...shared.records import PredictionRecord, RetrievedEvidence
from .client import RankifyWorkerClient
from .config import _worker_common

LOGGER = logging.getLogger(__name__)


def _rankify_index_root(spec) -> Path:
    return spec.index_root / spec.index_fingerprint()[:16] / "rankify" / spec.dataset


def _export_corpora(spec, bundle, config: ChunkingConfig) -> dict[str, Path]:
    chunker = DocumentChunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        min_chunk_chars=config.min_chunk_chars,
        parent_window_chars=config.parent_window_chars,
    )
    chunks = chunker.chunk_documents(bundle.documents)
    paths: dict[str, Path] = {}
    for language in _languages(spec.dataset):
        target = _rankify_index_root(spec) / language / "corpus.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(target, [
            {
                "id": chunk.id,
                "title": chunk.title,
                "text": chunk.text,
                "source_id": chunk.document_id,
                "source_path": chunk.source_path,
                "metadata": chunk.metadata,
            }
            for chunk in chunks if chunk.language == language
        ])
        paths[language] = target
    return paths


def _question_payload(question) -> dict[str, Any]:
    return {
        "id": question.id,
        "dataset": question.dataset,
        "language": question.language,
        "question": question.question,
        "split": question.split,
        "available": question.available,
        "source_ids": question.source_ids,
        "metadata": question.metadata,
    }


def _index_rankify(spec, bundle, worker: RankifyWorkerClient, chunking: ChunkingConfig) -> dict[str, Any]:
    corpora = _export_corpora(spec, bundle, chunking)
    languages = {}
    for language, corpus in corpora.items():
        request = {
            "schema_version": 1,
            "operation": "index",
            **_worker_common(spec),
            "corpus_path": str(corpus),
            "index_dir": str(corpus.parent / "indexes"),
        }
        languages[language] = worker.invoke(request)
        write_json(corpus.parent / "index-manifest.json", languages[language])
    return languages


def _convert_worker_rows(
    rows: list[dict], questions: list, technique: str, spec, corpus_fingerprint: str,
) -> list[dict]:
    by_id = {item.id: item for item in questions}
    converted = []
    seen = set()
    for row in rows:
        identifier = str(row.get("id"))
        if identifier not in by_id or identifier in seen:
            raise RuntimeError(f"Rankify returned an unknown or duplicate QA ID: {identifier}")
        seen.add(identifier)
        question = by_id[identifier]
        evidence = [RetrievedEvidence(
            source_id=str(item.get("source_id") or ""),
            chunk_id=str(item.get("chunk_id") or ""),
            text=str(item.get("text") or ""),
            rerank_score=float(item.get("score") or 0.0),
            metadata=dict(item.get("metadata") or {}),
        ) for item in row.get("evidence") or []]
        diagnostics = dict(row.get("diagnostics") or {})
        if question.source_ids:
            diagnostics["relevant_ranks"] = _relevant_ranks(evidence, question.source_ids)
            diagnostics["relevance_label"] = question.metadata.get("relevance_label")
        record = PredictionRecord(
            id=question.id,
            dataset=question.dataset,
            language=question.language,
            question=question.question,
            references=question.references,
            prediction=str(row.get("prediction") or ""),
            retrieved_source_ids=list(dict.fromkeys(item.source_id for item in evidence)),
            rerank_scores=[item.rerank_score for item in evidence],
            context_length=int(row.get("context_length") or 0),
            latency_ms=float(row.get("latency_ms") or 0.0),
            status=str(row.get("status") or "error"),
            available=question.available,
            error=row.get("error"),
            technique=technique,
            split=question.split,
            model_fingerprint=spec.generator.fingerprint(),
            corpus_fingerprint=corpus_fingerprint,
            decoding_config=spec.generator.decoding_dict(),
            gold_source_ids=question.source_ids,
            diagnostics=diagnostics,
        )
        converted.append(record.to_dict())
    missing = set(by_id) - seen
    if missing:
        raise RuntimeError(f"Rankify returned {len(rows)} of {len(questions)} QA records; missing {len(missing)}")
    return converted


def _evaluate_rankify(
    spec, bundle, worker: RankifyWorkerClient, output: Path, corpus_fingerprint: str,
) -> dict[str, Any]:
    questions = bundle.questions_for_split(spec.evaluation_split)
    results = {}
    with tempfile.TemporaryDirectory(prefix="rag-rankify-") as temporary:
        temp = Path(temporary)
        batches = []
        for language in _languages(spec.dataset):
            question_path = temp / f"questions-{language}.jsonl"
            write_jsonl(question_path, [
                _question_payload(item) for item in questions if item.language == language
            ])
            batches.append({
                "language": language,
                "corpus_path": str(_rankify_index_root(spec) / language / "corpus.jsonl"),
                "index_dir": str(_rankify_index_root(spec) / language / "indexes"),
                "questions_path": str(question_path),
            })

        for preset in spec.presets:
            raw_output = temp / f"{preset}.jsonl"
            try:
                response = worker.invoke({
                    "schema_version": 1,
                    "operation": "evaluate",
                    **_worker_common(spec),
                    "preset": preset,
                    "batches": batches,
                    "output_path": str(raw_output),
                })
            except Exception as exc:
                target = output / preset
                target.mkdir(parents=True, exist_ok=True)
                failure = {"status": "failed", "preset": preset, "error": f"{type(exc).__name__}: {exc}"}
                write_json(target / "worker-diagnostics.json", failure)
                results[preset] = failure
                LOGGER.error("Rankify preset %s failed during initialization: %s", preset, exc)
                continue
            rows = _convert_worker_rows(
                read_predictions(raw_output), questions, preset, spec, corpus_fingerprint,
            )
            target = output / preset
            target.mkdir(parents=True, exist_ok=True)
            write_jsonl(target / "predictions.jsonl", rows)
            report = summarize(rows)
            write_json(target / "metrics.json", report)
            write_json(target / "worker-diagnostics.json", response)
            results[preset] = report
    return results
