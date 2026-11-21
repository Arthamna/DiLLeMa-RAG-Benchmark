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
from .config import VARIANTS
from .embedding import BgeM3Embedding
from .pipeline import BgeM3Pipeline, candidate_hash, rank_candidates
from .store import BgeM3Store, ScoredCandidate

LOGGER = logging.getLogger(__name__)


def _release_gpu():
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


def _raise_if_fatal(exc):
    if is_fatal_accelerator_error(exc):
        raise exc


def index_root(spec, corpus_hash: str) -> Path:
    digest = spec.bge_m3.index_fingerprint(corpus_hash, ChunkingConfig())
    return spec.index_root / "bge_m3" / spec.dataset / digest[:16]


def run_index(spec, bundle) -> dict:
    config = spec.bge_m3
    if config is None:
        raise ValueError("BGE-M3 is not enabled")
    corpus_hash = _corpus_fingerprint(bundle.documents)
    digest = config.index_fingerprint(corpus_hash, ChunkingConfig())
    target = index_root(spec, corpus_hash)
    target.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device("BGE-M3 indexing")
    LOGGER.info("BGE-M3 indexing: loading %s at %s", config.model, config.revision)
    embedder = BgeM3Embedding(config, device)
    store = None
    try:
        write_json(target / "index-manifest.json", {"status": "building"})
        store = BgeM3Store(target, f"{spec.dataset}_{digest[:16]}")
        chunks = DocumentChunker(**asdict(ChunkingConfig())).chunk_documents(bundle.documents)
        collections = {}
        for language in _languages(spec.dataset):
            selected = [chunk for chunk in chunks if chunk.language == language]
            store.recreate(language, embedder.vector_size, embedder.multi_vector_size)
            for offset in range(0, len(selected), config.embedding_batch_size):
                batch = selected[offset:offset + config.embedding_batch_size]
                encoded = embedder.encode_passages([chunk.text for chunk in batch])
                store.add(language, batch, encoded)
                LOGGER.info("BGE-M3 index %s: %s/%s chunks",
                            language, offset + len(batch), len(selected))
            collections[language] = {
                "chunks": len(selected),
                "collection": store.collection(language),
            }
        manifest = {
            "status": "complete",
            "created_at": _now(),
            "technique": "bge_m3",
            "corpus_fingerprint": corpus_hash,
            "index_fingerprint": digest,
            "model_revision": embedder.revision,
            "requested_revision": config.revision,
            "config": config.public_dict(),
            "dense_dimension": embedder.vector_size,
            "multi_vector_dimension": embedder.multi_vector_size,
            "embedding_dtype": "float16" if config.use_fp16 and device.startswith("cuda") else "float32",
            "vectors": ["dense", "learned_sparse", "colbert"],
            "collections": collections,
            "indexing_is_model_training": False,
        }
        write_json(target / "index-manifest.json", manifest)
        return manifest
    finally:
        if store is not None:
            store.close()
        del embedder
        _release_gpu()


def _candidate_from_dict(value: dict) -> ScoredCandidate:
    return ScoredCandidate(
        RetrievedEvidence(**value["evidence"]),
        float(value["dense_score"]),
        float(value["sparse_score"]),
        float(value["multi_vector_score"]),
    )


def _retrieve(spec, questions, target: Path, manifest: dict, destination: Path, device: str):
    config = spec.bge_m3
    LOGGER.info("BGE-M3 retrieval: loading %s at %s", config.model, config.revision)
    embedder = BgeM3Embedding(config, device)
    LOGGER.info("BGE-M3 retrieval: model loaded; opening shared index")
    store = None
    try:
        if embedder.revision != manifest["model_revision"]:
            raise ValueError("BGE-M3 model revision changed; pin revision or rebuild the index")
        store = BgeM3Store(target, f"{spec.dataset}_{manifest['index_fingerprint'][:16]}")
        with destination.open("w", encoding="utf-8") as handle:
            for number, question in enumerate(questions, 1):
                started = time.perf_counter()
                candidates, error = [], None
                if question.available:
                    try:
                        query = embedder.encode_query(question.question)
                        candidates = store.search(question.language, query, spec.candidates)
                    except RuntimeError:
                        # Runtime failures are model/runtime failures, not bad
                        # questions. Preserve their traceback in benchmark.log.
                        raise
                    except Exception as exc:
                        _raise_if_fatal(exc)
                        error = f"{type(exc).__name__}: {exc}"
                row = {
                    "id": question.id,
                    "question": question.question,
                    "language": question.language,
                    "candidates": [asdict(item) for item in candidates],
                    "error": error,
                    "candidate_pool_hash": candidate_hash(candidates),
                    "retrieval_latency_ms": (time.perf_counter() - started) * 1000,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                LOGGER.info("BGE-M3 retrieval %s/%s: %s (%s candidates)",
                            number, len(questions), question.id, len(candidates))
    finally:
        if store is not None:
            store.close()
        del embedder
        _release_gpu()


def _delta(first: dict, second: dict) -> dict:
    values = {
        key: second[key] - first[key]
        for key in ("exact_match", "token_f1", "error_rate")
    }
    values.update({
        key: second[key]["score"] - first[key]["score"]
        for key in ("BLEU_raw", "BLEU_content")
    })
    for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"):
        baseline, candidate = first["retrieval"][key], second["retrieval"][key]
        values[key] = candidate - baseline if baseline is not None and candidate is not None else None
    values["latency_mean_ms"] = second["average_latency_ms"] - first["average_latency_ms"]
    values["cpu_fallback_count"] = second["cpu_fallback_count"] - first["cpu_fallback_count"]
    values["latency_ms"] = {
        stage: {key: second["latency_ms"][stage][key] - stats[key] for key in stats}
        for stage, stats in first["latency_ms"].items()
    }
    return values


def _ablation(reports, rows_by_variant):
    pool_signatures = {
        name: [(row["id"], row["diagnostics"]["candidate_pool_hash"]) for row in rows]
        for name, rows in rows_by_variant.items()
    }
    baseline_signature = pool_signatures[VARIANTS[0]]
    if any(signature != baseline_signature for signature in pool_signatures.values()):
        raise RuntimeError("BGE-M3 ablation candidate pools differ")
    groups = {name: _summary_groups(reports[name]) for name in VARIANTS}
    comparisons = {}
    for variant in VARIANTS[1:]:
        comparisons[variant] = {
            language: _delta(groups[VARIANTS[0]][language], group)
            for language, group in groups[variant].items()
        }
    return {
        "baseline": VARIANTS[0],
        "candidate_pools_identical": True,
        "variants": list(VARIANTS),
        "variant_minus_dense": comparisons,
    }


def run_evaluate(spec, bundle, output: Path, variants=VARIANTS) -> dict:
    config = spec.bge_m3
    if config is None:
        raise ValueError("BGE-M3 is not enabled")
    if not variants or any(name not in VARIANTS for name in variants):
        raise ValueError("Unsupported BGE-M3 variant")
    questions = bundle.questions_for_split(spec.evaluation_split)
    corpus_hash = _corpus_fingerprint(bundle.documents)
    target = index_root(spec, corpus_hash)
    manifest_path = target / "index-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"BGE-M3 index missing: {manifest_path}; run --stage index or all")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = config.index_fingerprint(corpus_hash, ChunkingConfig())
    if manifest.get("status") != "complete" or manifest.get("index_fingerprint") != expected:
        raise ValueError("BGE-M3 index is incomplete or incompatible; rebuild it")
    device = resolve_torch_device("BGE-M3 evaluation")
    reports, all_rows = {}, {}
    with tempfile.TemporaryDirectory(prefix="rag-bge-m3-") as temporary:
        candidates_path = Path(temporary) / "candidates.jsonl"
        _retrieve(spec, questions, target, manifest, candidates_path, device)
        LOGGER.info("BGE-M3 generation: retrieval model released; loading shared generator")
        runtime = spec.generator.runtime()
        try:
            for variant in variants:
                destination = output / variant
                destination.mkdir(parents=True, exist_ok=True)
                rows = []
                with candidates_path.open(encoding="utf-8") as inputs, (
                    destination / "predictions.jsonl"
                ).open("w", encoding="utf-8") as predictions:
                    for number, (question, line) in enumerate(zip(questions, inputs), 1):
                        row = json.loads(line)
                        if row["id"] != question.id:
                            raise RuntimeError("BGE-M3 prepared QA order changed")
                        pipeline = BgeM3Pipeline(row, runtime, spec.top_k, variant, config.weights)
                        record = evaluate_questions(
                            [question], pipeline, spec.dataset, variant,
                            spec.generator.fingerprint(), corpus_hash,
                            spec.generator.decoding_dict(),
                        )[0]
                        candidates = [_candidate_from_dict(item) for item in row["candidates"]]
                        selected = (
                            rank_candidates(candidates, variant, config.weights)[:spec.top_k]
                            if question.available and not row.get("error") else []
                        )
                        record.retrieved_source_ids = list(dict.fromkeys(
                            item.source_id for item in selected
                        ))
                        record.rerank_scores = [item.rerank_score for item in selected]
                        if question.source_ids:
                            ranks = _relevant_ranks(selected, question.source_ids)
                            record.diagnostics.update(
                                relevant_ranks=ranks,
                                source_recall_at_10=float(bool(ranks)),
                            )
                        retrieval_ms = float(row["retrieval_latency_ms"])
                        record.diagnostics.update({
                            "candidate_pool_hash": row["candidate_pool_hash"],
                            "candidate_chunk_ids": [item.evidence.chunk_id for item in candidates],
                            "candidate_cutoff": spec.candidates * 2,
                            "candidates_per_mode": spec.candidates,
                            "candidate_pool_limit": spec.candidates * 2,
                            "candidate_relevant_ranks": _relevant_ranks(
                                [item.evidence for item in candidates], question.source_ids,
                                limit=len(candidates),
                            ),
                            "retrieved_candidates": len(candidates),
                            "candidate_modes": ["dense", "learned_sparse"],
                            "component_scores": {
                                item.evidence.chunk_id: {
                                    "dense": item.dense_score,
                                    "sparse": item.sparse_score,
                                    "multi_vector": item.multi_vector_score,
                                } for item in candidates
                            },
                            "fusion_weights": list(config.weights),
                            "device": device,
                            "gpu_fallback": (
                                device == "cpu" and os.getenv("RAG_DEVICE", "auto").casefold() != "cpu"
                            ),
                            "retrieval_latency_ms": retrieval_ms,
                            "pipeline_fingerprint": config.pipeline_fingerprint(
                                corpus_hash, ChunkingConfig(), spec.candidates, spec.top_k,
                            ),
                            "index_fingerprint": manifest["index_fingerprint"],
                            "model_revision": manifest["model_revision"],
                            "embedding_dtype": manifest["embedding_dtype"],
                        })
                        record.latency_ms = round(record.latency_ms + retrieval_ms, 3)
                        result = record.to_dict()
                        rows.append(result)
                        predictions.write(json.dumps(result, ensure_ascii=False) + "\n")
                        predictions.flush()
                        LOGGER.info("%s generation %s/%s: %s (%s)",
                                    variant, number, len(questions), question.id, record.status)
                if len(rows) != len(questions):
                    raise RuntimeError("BGE-M3 did not produce one record per scheduled QA")
                all_rows[variant] = rows
                reports[variant] = summarize(rows)
                write_json(destination / "metrics.json", reports[variant])
                write_json(destination / "bleu-summary.json", reports[variant])
                write_jsonl(destination / "missing-context.jsonl", [
                    {"id": item.id, "language": item.language, "source_ids": item.source_ids}
                    for item in questions if not item.available
                ])
                write_json(destination / "manifest.json", {
                    "technique": variant,
                    "created_at": _now(),
                    "config": config.public_dict(),
                    "generator": spec.generator.public_dict(),
                    "split": spec.evaluation_split,
                    "scheduled_questions": len(questions),
                    "prediction_records": len(rows),
                    "corpus_fingerprint": corpus_hash,
                    "model_fingerprint": spec.generator.fingerprint(),
                    "decoding_config": spec.generator.decoding_dict(),
                    "index": manifest,
                    "pipeline_fingerprint": config.pipeline_fingerprint(
                        corpus_hash, ChunkingConfig(), spec.candidates, spec.top_k,
                    ),
                    "retrieval": {
                        "candidates_per_mode": spec.candidates,
                        "candidate_pool_limit": spec.candidates * 2,
                        "top_k": spec.top_k,
                        "candidate_modes": ["dense", "learned_sparse"],
                    },
                    "device": device,
                    "latency_definition": "per-QA stage sum; excludes model loading and batch queue wait",
                })
                write_json(destination / "diagnostics.json", {
                    "device": device,
                    "shared_candidate_retrieval": True,
                    "candidate_pools_identical_by_construction": True,
                    "fusion_weights": list(config.weights),
                })
        finally:
            if "pipeline" in locals():
                del pipeline
            del runtime
            _release_gpu()
    if all(name in reports for name in VARIANTS):
        write_json(output / "bge-m3-ablation.json", _ablation(reports, all_rows))
    return reports
