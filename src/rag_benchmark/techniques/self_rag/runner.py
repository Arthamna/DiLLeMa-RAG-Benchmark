from __future__ import annotations

import gc
import json
import logging
import os
import sys
import tempfile
import time
from contextlib import ExitStack
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
from .generator import SelfRagLLM
from .pipeline import PreparedAnswer, SelfRagPipeline
from .retriever import MContriever, passage_text
from .store import SelfRagStore

LOGGER = logging.getLogger(__name__)
LATENCY_DEFINITION = "per-QA retrieval plus variant generation stages; excludes model loading"


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


def _require(spec):
    if spec.self_rag is None:
        raise ValueError("Self-RAG is not enabled")
    return spec.self_rag


def index_root(spec, corpus_hash: str) -> Path:
    digest = spec.self_rag.index_fingerprint(corpus_hash, ChunkingConfig())
    return spec.index_root / "self_rag" / spec.dataset / digest[:16]


def run_index(spec, bundle) -> dict:
    config = _require(spec)
    corpus_hash = _corpus_fingerprint(bundle.documents)
    digest = config.index_fingerprint(corpus_hash, ChunkingConfig())
    target = index_root(spec, corpus_hash)
    target.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device("Self-RAG indexing")
    LOGGER.info("Self-RAG indexing: loading %s at %s", config.retriever_model, config.retriever_revision)
    encoder = MContriever(config, device)
    store = None
    try:
        write_json(target / "index-manifest.json", {"status": "building"})
        store = SelfRagStore(target, f"{spec.dataset}_{digest[:16]}")
        chunks = DocumentChunker(**asdict(ChunkingConfig())).chunk_documents(bundle.documents)
        collections = {}
        for language in _languages(spec.dataset):
            selected = [chunk for chunk in chunks if chunk.language == language]
            store.recreate(language, encoder.vector_size)
            for offset in range(0, len(selected), config.embedding_batch_size):
                batch = selected[offset:offset + config.embedding_batch_size]
                store.add(language, batch, encoder.encode_passages(
                    [passage_text(chunk.title, chunk.text) for chunk in batch],
                ))
                LOGGER.info("Self-RAG index %s: %s/%s chunks", language, offset + len(batch), len(selected))
            collections[language] = {"chunks": len(selected), "collection": store.collection(language)}
        manifest = {
            "status": "complete",
            "created_at": _now(),
            "technique": "self_rag",
            "corpus_fingerprint": corpus_hash,
            "index_fingerprint": digest,
            "retriever_model": config.retriever_model,
            "retriever_revision": encoder.revision,
            "requested_retriever_revision": config.retriever_revision,
            "vector_dimension": encoder.vector_size,
            "collections": collections,
            "indexing_is_model_training": False,
        }
        write_json(target / "index-manifest.json", manifest)
        return manifest
    finally:
        if store is not None:
            store.close()
        del encoder
        _release_gpu()


def _retrieve(spec, questions, target: Path, manifest: dict, destination: Path, device: str):
    config = spec.self_rag
    LOGGER.info("Self-RAG retrieval: loading %s", config.retriever_model)
    encoder = MContriever(config, device)
    store = None
    try:
        if encoder.revision != manifest["retriever_revision"]:
            raise ValueError("mContriever revision changed; pin retriever_revision or rebuild the index")
        store = SelfRagStore(target, f"{spec.dataset}_{manifest['index_fingerprint'][:16]}")
        with destination.open("w", encoding="utf-8") as handle:
            for number, question in enumerate(questions, 1):
                started = time.perf_counter()
                candidates, error = [], None
                if question.available:
                    try:
                        candidates = store.search(
                            question.language, encoder.encode_query(question.question), spec.candidates,
                        )
                    except RuntimeError:
                        raise
                    except Exception as exc:
                        _raise_if_fatal(exc)
                        error = f"{type(exc).__name__}: {exc}"
                handle.write(json.dumps({
                    "id": question.id,
                    "candidates": [asdict(item) for item in candidates],
                    "error": error,
                    "retrieval_latency_ms": (time.perf_counter() - started) * 1000,
                }, ensure_ascii=False) + "\n")
                LOGGER.info("Self-RAG retrieval %s/%s: %s (%s candidates)",
                            number, len(questions), question.id, len(candidates))
    finally:
        if store is not None:
            store.close()
        del encoder
        _release_gpu()


def _delta(first: dict, second: dict) -> dict:
    values = {key: second[key] - first[key] for key in ("exact_match", "token_f1", "error_rate")}
    values.update({key: second[key]["score"] - first[key]["score"] for key in ("BLEU_raw", "BLEU_content")})
    for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"):
        baseline, candidate = first["retrieval"][key], second["retrieval"][key]
        values[key] = candidate - baseline if baseline is not None and candidate is not None else None
    values["latency_mean_ms"] = second["average_latency_ms"] - first["average_latency_ms"]
    return values


def _ablation(reports: dict, rows: dict) -> dict:
    groups = {name: _summary_groups(reports[name]) for name in VARIANTS}
    adaptive = [row for row in rows["self_rag_adaptive"] if row["status"] != "missing_context"]
    retrieval_rate = {}
    for language in groups["self_rag_adaptive"]:
        selected = [row for row in adaptive if row["language"] == language]
        retrieval_rate[language] = (
            sum(bool(row["diagnostics"].get("retrieval_used")) for row in selected) / len(selected)
            if selected else None
        )
    return {
        "baseline": "self_rag_always",
        "variants": list(VARIANTS),
        "adaptive_retrieval_rate": retrieval_rate,
        "adaptive_minus_always": {
            language: _delta(groups["self_rag_always"][language], group)
            for language, group in groups["self_rag_adaptive"].items()
        },
    }


def _load_manifest(spec, corpus_hash: str) -> tuple[Path, dict]:
    target = index_root(spec, corpus_hash)
    path = target / "index-manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Self-RAG index missing: {path}; run --stage index or all")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = spec.self_rag.index_fingerprint(corpus_hash, ChunkingConfig())
    if manifest.get("status") != "complete" or manifest.get("index_fingerprint") != expected:
        raise ValueError("Self-RAG index is incomplete or incompatible; rebuild it")
    return target, manifest


def run_evaluate(spec, bundle, output: Path, variants=VARIANTS) -> dict:
    config = _require(spec)
    variants = tuple(variants)
    if not variants or any(name not in VARIANTS for name in variants):
        raise ValueError("Unsupported Self-RAG variant")
    if config.ndocs > spec.candidates:
        raise ValueError("self_rag.ndocs cannot exceed retrieval.candidates")
    questions = bundle.questions_for_split(spec.evaluation_split)
    corpus_hash = _corpus_fingerprint(bundle.documents)
    target, manifest = _load_manifest(spec, corpus_hash)
    device = resolve_torch_device("Self-RAG evaluation")
    pipeline_fingerprint = config.pipeline_fingerprint(corpus_hash, ChunkingConfig(), spec.candidates)
    model_fingerprint = config.model_fingerprint()
    gpu_fallback = device == "cpu" and os.getenv("RAG_DEVICE", "auto").casefold() != "cpu"
    rows: dict[str, list[dict]] = {name: [] for name in variants}
    with tempfile.TemporaryDirectory(prefix="rag-self-rag-") as temporary:
        candidates_path = Path(temporary) / "candidates.jsonl"
        _retrieve(spec, questions, target, manifest, candidates_path, device)
        LOGGER.info("Self-RAG generation: retriever released; loading %s", config.model)
        llm = SelfRagLLM(config, device)
        try:
            pipeline = SelfRagPipeline(llm, config)
            with ExitStack() as stack:
                handles = {}
                for name in variants:
                    (output / name).mkdir(parents=True, exist_ok=True)
                    handles[name] = stack.enter_context(
                        (output / name / "predictions.jsonl").open("w", encoding="utf-8"),
                    )
                inputs = stack.enter_context(candidates_path.open(encoding="utf-8"))
                for number, (question, line) in enumerate(zip(questions, inputs), 1):
                    row = json.loads(line)
                    if row["id"] != question.id:
                        raise RuntimeError("Self-RAG prepared QA order changed")
                    candidates = [RetrievedEvidence(**item) for item in row["candidates"]]
                    outcomes, error = {}, row.get("error")
                    if question.available and not error:
                        try:
                            outcomes = pipeline.run(question.question, candidates, variants)
                        except Exception as exc:
                            _raise_if_fatal(exc)
                            error = f"{type(exc).__name__}: {exc}"
                    retrieval_ms = float(row["retrieval_latency_ms"])
                    for name in variants:
                        outcome = outcomes.get(name)
                        record = evaluate_questions(
                            [question], PreparedAnswer(outcome, error), spec.dataset, name,
                            model_fingerprint, corpus_hash, config.decoding_dict(),
                        )[0]
                        record.diagnostics.update({
                            "candidate_cutoff": spec.candidates,
                            "candidate_chunk_ids": [item.chunk_id for item in candidates],
                            "candidate_relevant_ranks": _relevant_ranks(
                                candidates, question.source_ids, limit=len(candidates),
                            ),
                            "retrieved_candidates": len(candidates),
                            "ndocs": config.ndocs,
                            "retrieval_latency_ms": retrieval_ms,
                            "device": device,
                            "gpu_fallback": gpu_fallback,
                            "pipeline_fingerprint": pipeline_fingerprint,
                            "index_fingerprint": manifest["index_fingerprint"],
                            "retriever_revision": manifest["retriever_revision"],
                            "model_revision": llm.revision,
                        })
                        generation_ms = outcome.generation_ms if outcome is not None else 0.0
                        record.latency_ms = round(retrieval_ms + generation_ms, 3)
                        result = record.to_dict()
                        rows[name].append(result)
                        handles[name].write(json.dumps(result, ensure_ascii=False) + "\n")
                        handles[name].flush()
                    LOGGER.info("Self-RAG generation %s/%s: %s", number, len(questions), question.id)
        finally:
            if "pipeline" in locals():
                del pipeline
            revision, quantization, dtype = llm.revision, llm.quantization, llm.dtype
            del llm
            _release_gpu()
    reports = {}
    for name in variants:
        if len(rows[name]) != len(questions):
            raise RuntimeError("Self-RAG did not produce one record per scheduled QA")
        destination = output / name
        reports[name] = summarize(rows[name])
        write_json(destination / "metrics.json", reports[name])
        write_json(destination / "bleu-summary.json", reports[name])
        write_jsonl(destination / "missing-context.jsonl", [
            {"id": item.id, "language": item.language, "source_ids": item.source_ids}
            for item in questions if not item.available
        ])
        write_json(destination / "manifest.json", {
            "technique": name,
            "created_at": _now(),
            "config": config.public_dict(),
            "generator": {**config.generator_dict(), "resolved_revision": revision,
                          "effective_quantization": quantization, "effective_dtype": dtype},
            "split": spec.evaluation_split,
            "scheduled_questions": len(questions),
            "prediction_records": len(rows[name]),
            "corpus_fingerprint": corpus_hash,
            "model_fingerprint": model_fingerprint,
            "decoding_config": config.decoding_dict(),
            "index": manifest,
            "pipeline_fingerprint": pipeline_fingerprint,
            "retrieval": {"candidates": spec.candidates, "ndocs": config.ndocs},
            "device": device,
            "latency_definition": LATENCY_DEFINITION,
        })
        write_json(destination / "diagnostics.json", {
            "device": device,
            "gpu_fallback": gpu_fallback,
            "shared_generation_across_variants": len(variants) > 1,
            "retrieval_probe_max_new_tokens": 1,
            "threshold": config.threshold,
            "weights": {"w_rel": config.w_rel, "w_sup": config.w_sup, "w_use": config.w_use},
        })
    if all(name in reports for name in VARIANTS):
        write_json(output / "self-rag-ablation.json", _ablation(reports, rows))
    return reports
