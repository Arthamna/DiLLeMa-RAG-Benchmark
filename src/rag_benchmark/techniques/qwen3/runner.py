from __future__ import annotations

import gc
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from ...shared.artifacts import _corpus_fingerprint, _languages, _now
from ...shared.chunking import DocumentChunker
from ...shared.comparison import _summary_groups
from ...shared.config import ChunkingConfig
from ...shared.device import is_fatal_accelerator_error, resolve_torch_device
from ...shared.evaluation import _relevant_ranks, evaluate_questions
from ...shared.metrics import summarize, write_json, write_jsonl
from ...shared.records import RetrievedEvidence
from .config import VARIANTS, fingerprint
from .embedding import Qwen3Embedding
from .pipeline import Qwen3Pipeline, candidate_hash, rerank_candidates
from .reranker import Qwen3Reranker
from .store import DenseStore

LOGGER = logging.getLogger(__name__)


def _release_gpu():
    # Do not mask an original CUDA failure with a secondary cleanup failure.
    failed = sys.exc_info()[0] is not None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass
    except Exception:
        if not failed:
            raise


def index_root(spec, corpus_hash: str) -> Path:
    digest = spec.qwen3.index_fingerprint(corpus_hash, ChunkingConfig())
    return spec.index_root / "qwen3" / spec.dataset / digest[:16]


def run_index(spec, bundle) -> dict:
    config = spec.qwen3
    corpus_hash = _corpus_fingerprint(bundle.documents)
    digest = config.index_fingerprint(corpus_hash, ChunkingConfig())
    target = index_root(spec, corpus_hash)
    target.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device("Qwen3 indexing")
    LOGGER.info("Qwen3 indexing: loading %s", config.embedding_model)
    embedder = Qwen3Embedding(config, device)
    store = None
    try:
        write_json(target / "index-manifest.json", {"status": "building"})
        store = DenseStore(target, f"{spec.dataset}_{digest[:16]}")
        chunks = DocumentChunker(**asdict(ChunkingConfig())).chunk_documents(bundle.documents)
        collections = {}
        for language in _languages(spec.dataset):
            selected = [chunk for chunk in chunks if chunk.language == language]
            store.recreate(language, embedder.vector_size)
            for offset in range(0, len(selected), config.embedding_batch_size):
                batch = selected[offset:offset + config.embedding_batch_size]
                store.add(language, batch, embedder.encode_passages([chunk.text for chunk in batch]))
                LOGGER.info("Qwen3 index %s: %s/%s chunks", language, offset + len(batch), len(selected))
            collections[language] = {"chunks": len(selected), "collection": store.collection(language)}
        manifest = {
            "status": "complete", "created_at": _now(), "technique": "qwen3",
            "corpus_fingerprint": corpus_hash, "index_fingerprint": digest,
            "embedding_revision": embedder.revision, "config": config.public_dict(),
            "embedding_dtype": config.torch_dtype if device.startswith("cuda") else "float32",
            "collections": collections, "indexing_is_model_training": False,
        }
        write_json(target / "index-manifest.json", manifest)
        return manifest
    finally:
        if store is not None:
            store.close()
        del embedder
        _release_gpu()


def _raise_if_fatal(exc):
    if is_fatal_accelerator_error(exc):
        raise exc


def _retrieve(spec, questions, target, manifest, destination, device):
    LOGGER.info("Qwen3 retrieval: loading %s", spec.qwen3.embedding_model)
    embedder = Qwen3Embedding(spec.qwen3, device)
    store = None
    try:
        if embedder.revision != manifest["embedding_revision"]:
            raise ValueError("Qwen3 embedding revision changed; pin revision or rebuild the index")
        store = DenseStore(target, f"{spec.dataset}_{manifest['index_fingerprint'][:16]}")
        with destination.open("w", encoding="utf-8") as handle:
            for number, question in enumerate(questions, 1):
                started = time.perf_counter()
                evidence, error = [], None
                if question.available:
                    try:
                        vector = embedder.encode_query(question.question)
                        evidence = store.search(question.language, vector, spec.candidates)
                    except Exception as exc:
                        _raise_if_fatal(exc)
                        error = f"{type(exc).__name__}: {exc}"
                row = {
                    "id": question.id, "question": question.question, "language": question.language,
                    "candidates": [asdict(item) for item in evidence], "error": error,
                    "candidate_pool_hash": candidate_hash(evidence),
                    "retrieval_latency_ms": (time.perf_counter() - started) * 1000,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                LOGGER.info("Qwen3 retrieval %s/%s: %s (%s candidates)",
                            number, len(questions), question.id, len(evidence))
    finally:
        if store is not None:
            store.close()
        del embedder
        _release_gpu()


def _rerank(spec, source, destination, device):
    LOGGER.info("Qwen3 reranking: loading %s (%s)",
                spec.qwen3.reranker_model, spec.qwen3.reranker_quantization)
    reranker = Qwen3Reranker(spec.qwen3, device)
    try:
        with source.open(encoding="utf-8") as inputs, destination.open("w", encoding="utf-8") as output:
            for number, line in enumerate(inputs, 1):
                row = json.loads(line)
                started = time.perf_counter()
                row["ranked"] = []
                row["rerank_error"] = None
                try:
                    candidates = [RetrievedEvidence(**item) for item in row["candidates"]]
                    row["ranked"] = [asdict(item) for item in rerank_candidates(
                        row["question"], candidates, reranker,
                    )] if candidates else []
                except Exception as exc:
                    _raise_if_fatal(exc)
                    row["rerank_error"] = f"{type(exc).__name__}: {exc}"
                row["rerank_latency_ms"] = (time.perf_counter() - started) * 1000
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                LOGGER.info("Qwen3 reranking %s: %s", number, row["id"])
        return {
            "revision": reranker.revision, "quantization": reranker.quantization,
            "torch_dtype": spec.qwen3.torch_dtype if device.startswith("cuda") else "float32",
        }
    finally:
        del reranker
        _release_gpu()


def _ablation(reports, rows_by_variant):
    baseline, candidate = (rows_by_variant[name] for name in VARIANTS)
    if [(row["id"], row["diagnostics"]["candidate_pool_hash"]) for row in baseline] != [
        (row["id"], row["diagnostics"]["candidate_pool_hash"]) for row in candidate
    ]:
        raise RuntimeError("Qwen3 ablation candidate pools differ")
    groups = {name: _summary_groups(reports[name]) for name in VARIANTS}
    deltas = {}
    for language, first in groups[VARIANTS[0]].items():
        second = groups[VARIANTS[1]][language]
        values = {
            key: second[key] - first[key] for key in ("exact_match", "token_f1", "error_rate")
        }
        values.update({key: second[key]["score"] - first[key]["score"]
                       for key in ("BLEU_raw", "BLEU_content")})
        for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"):
            a, b = first["retrieval"][key], second["retrieval"][key]
            values[key] = b - a if a is not None and b is not None else None
        values["latency_mean_ms"] = second["average_latency_ms"] - first["average_latency_ms"]
        values["cpu_fallback_count"] = second["cpu_fallback_count"] - first["cpu_fallback_count"]
        if "candidate_recall_at_100" in first["retrieval"]:
            a, b = first["retrieval"]["candidate_recall_at_100"], second["retrieval"]["candidate_recall_at_100"]
            values["candidate_recall_at_100"] = b - a if a is not None and b is not None else None
        values["latency_ms"] = {
            stage: {key: second["latency_ms"][stage][key] - stats[key] for key in stats}
            for stage, stats in first["latency_ms"].items()
        }
        deltas[language] = values
    return {"candidate_pools_identical": True, "candidate_minus_baseline": deltas,
            "baseline": VARIANTS[0], "candidate": VARIANTS[1]}


def run_evaluate(spec, bundle, output: Path, variants=VARIANTS) -> dict:
    config = spec.qwen3
    if not variants or any(name not in VARIANTS for name in variants):
        raise ValueError("Unsupported Qwen3 variant")
    questions = bundle.questions_for_split(spec.evaluation_split)
    corpus_hash = _corpus_fingerprint(bundle.documents)
    target = index_root(spec, corpus_hash)
    manifest_path = target / "index-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Qwen3 index missing: {manifest_path}; run --stage index or all")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete"
            or manifest.get("index_fingerprint") != config.index_fingerprint(corpus_hash, ChunkingConfig())):
        raise ValueError("Qwen3 index is incomplete or incompatible; rebuild it")
    device = resolve_torch_device("Qwen3 evaluation")
    reports, all_rows = {}, {}
    with tempfile.TemporaryDirectory(prefix="rag-qwen3-") as temporary:
        candidates = Path(temporary) / "candidates.jsonl"
        prepared = Path(temporary) / "prepared.jsonl"
        _retrieve(spec, questions, target, manifest, candidates, device)
        reranker_info = {}
        if VARIANTS[1] in variants:
            reranker_info = _rerank(spec, candidates, prepared, device)
        else:
            prepared = candidates
        LOGGER.info("Qwen3 generation: retrieval models released; loading shared generator")
        runtime = spec.generator.runtime()
        try:
            for variant in variants:
                destination = output / variant
                destination.mkdir(parents=True, exist_ok=True)
                rows = []
                with prepared.open(encoding="utf-8") as inputs, (destination / "predictions.jsonl").open(
                    "w", encoding="utf-8",
                ) as predictions:
                    for number, (question, line) in enumerate(zip(questions, inputs), 1):
                        row = json.loads(line)
                        if row["id"] != question.id:
                            raise RuntimeError("Qwen3 prepared QA order changed")
                        reranked = variant == VARIANTS[1]
                        pipeline = Qwen3Pipeline(row, runtime, spec.top_k, reranked)
                        record = evaluate_questions(
                            [question], pipeline, spec.dataset, variant, spec.generator.fingerprint(),
                            corpus_hash, spec.generator.decoding_dict(),
                        )[0]
                        record.diagnostics.update(pipeline.diagnostics)
                        retrieval_ms = row["retrieval_latency_ms"]
                        rerank_ms = row.get("rerank_latency_ms", 0) if reranked else 0
                        evidence = [RetrievedEvidence(**item) for item in row["candidates"]]
                        # Retrieval metrics remain measurable even if generation fails.
                        selected = [RetrievedEvidence(**item) for item in row.get(
                            "ranked" if reranked else "candidates", [],
                        )[:spec.top_k]] if question.available and not row.get("error") else []
                        if question.available and row.get("rerank_error") and reranked:
                            selected = []
                        record.retrieved_source_ids = list(dict.fromkeys(item.source_id for item in selected))
                        record.rerank_scores = [item.rerank_score for item in selected]
                        if question.source_ids:
                            ranks = _relevant_ranks(selected, question.source_ids)
                            record.diagnostics.update(relevant_ranks=ranks, source_recall_at_10=float(bool(ranks)))
                        record.diagnostics.update({
                            "candidate_pool_hash": row["candidate_pool_hash"],
                            "candidate_chunk_ids": [item.chunk_id for item in evidence],
                            "candidate_cutoff": spec.candidates,
                            "candidate_relevant_ranks": _relevant_ranks(evidence, question.source_ids, limit=spec.candidates),
                            "retrieved_candidates": len(evidence), "device": device,
                            "gpu_fallback": device == "cpu" and os.getenv("RAG_DEVICE", "auto") != "cpu",
                            "retrieval_latency_ms": retrieval_ms, "rerank_latency_ms": rerank_ms,
                            "pipeline_fingerprint": fingerprint({
                                "config": config.public_dict(), "candidates": spec.candidates,
                                "top_k": spec.top_k, "prompt_version": 1,
                            }),
                            "embedding_revision": manifest["embedding_revision"],
                            "embedding_dtype": config.torch_dtype if device.startswith("cuda") else "float32",
                            "reranker": reranker_info if reranked else None,
                        })
                        record.latency_ms = round(record.latency_ms + retrieval_ms + rerank_ms, 3)
                        result = record.to_dict()
                        rows.append(result)
                        predictions.write(json.dumps(result, ensure_ascii=False) + "\n")
                        predictions.flush()
                        LOGGER.info("%s generation %s/%s: %s (%s)",
                                    variant, number, len(questions), question.id, record.status)
                if len(rows) != len(questions):
                    raise RuntimeError("Qwen3 did not produce one record per scheduled QA")
                all_rows[variant] = rows
                reports[variant] = summarize(rows)
                write_json(destination / "metrics.json", reports[variant])
                write_json(destination / "bleu-summary.json", reports[variant])
                write_jsonl(destination / "missing-context.jsonl", [
                    {"id": item.id, "language": item.language, "source_ids": item.source_ids}
                    for item in questions if not item.available
                ])
                write_json(destination / "manifest.json", {
                    "technique": variant, "created_at": _now(), "config": config.public_dict(),
                    "generator": spec.generator.public_dict(), "split": spec.evaluation_split,
                    "scheduled_questions": len(questions), "prediction_records": len(rows),
                    "corpus_fingerprint": corpus_hash, "model_fingerprint": spec.generator.fingerprint(),
                    "decoding_config": spec.generator.decoding_dict(), "index": manifest,
                    "retrieval": {"candidates": spec.candidates, "top_k": spec.top_k},
                    "device": device, "reranker": reranker_info if variant == VARIANTS[1] else None,
                    "latency_definition": "per-QA stage sum; excludes model loading and batch queue wait",
                })
                write_json(destination / "diagnostics.json", {
                    "device": device, "reranker": reranker_info if variant == VARIANTS[1] else None,
                    "shared_candidate_retrieval": True,
                })
        finally:
            # Pipeline also holds the runtime: release it before allocating another model.
            if "pipeline" in locals():
                del pipeline
            del runtime
            _release_gpu()
    if all(name in reports for name in VARIANTS):
        write_json(output / "qwen3-ablation.json", _ablation(reports, all_rows))
    return reports
