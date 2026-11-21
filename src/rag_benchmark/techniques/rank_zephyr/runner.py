from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

from ...shared.artifacts import _corpus_fingerprint, _languages, _now
from ...shared.chunking import DocumentChunker
from ...shared.config import ChunkingConfig
from ...shared.device import is_fatal_accelerator_error, resolve_torch_device
from ...shared.evaluation import _relevant_ranks, evaluate_questions
from ...shared.metrics import summarize, write_json, write_jsonl
from ...shared.records import RetrievedEvidence, TechniqueResult
from .config import VARIANTS, fingerprint
from .retrieval import DenseEmbedding, DenseStore


LOGGER = logging.getLogger(__name__)


def _release_gpu(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and sys.exc_info()[0] is None:
        import torch

        torch.cuda.empty_cache()


def _index_location(spec, corpus_hash: str, revision: str) -> tuple[str, Path]:
    digest = spec.rank_zephyr.index_fingerprint(corpus_hash, revision)
    return digest, spec.index_root / "rank_zephyr" / spec.dataset / digest


def run_index(spec, bundle) -> dict:
    config = spec.rank_zephyr
    corpus_hash = _corpus_fingerprint(bundle.documents)
    device = resolve_torch_device("RankZephyr indexing")
    LOGGER.info("RankZephyr indexing: loading %s", config.embedding_model)
    embedding = DenseEmbedding(config, device)
    try:
        digest, target = _index_location(spec, corpus_hash, embedding.revision)
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "index-manifest.json"
        write_json(manifest_path, {"status": "building"})
        store = DenseStore(target, f"{spec.dataset}_{digest[:16]}")
        try:
            chunks = DocumentChunker(**asdict(ChunkingConfig())).chunk_documents(bundle.documents)
            collections = {}
            for language in _languages(spec.dataset):
                selected = [chunk for chunk in chunks if chunk.language == language]
                store.recreate(language, embedding.dimension)
                for offset in range(0, len(selected), config.embedding_batch_size):
                    batch = selected[offset:offset + config.embedding_batch_size]
                    store.add(language, batch, embedding.encode([chunk.text for chunk in batch]))
                    LOGGER.info("RankZephyr index %s: %s/%s", language, offset + len(batch), len(selected))
                collections[language] = {"name": store.collection(language), "chunks": len(selected)}
            manifest = {
                "status": "complete", "created_at": _now(), "technique": "rank_zephyr",
                "corpus_fingerprint": corpus_hash, "index_fingerprint": digest,
                "embedding_model": config.embedding_model, "embedding_revision": embedding.revision,
                "embedding_max_length": config.embedding_max_length, "collections": collections,
            }
            write_json(manifest_path, manifest)
            return manifest
        finally:
            store.close()
    finally:
        del embedding
        _release_gpu(device)


def _retrieve(spec, questions, corpus_hash: str, destination: Path, device: str) -> dict:
    LOGGER.info("RankZephyr retrieval: loading %s", spec.rank_zephyr.embedding_model)
    embedding = DenseEmbedding(spec.rank_zephyr, device)
    try:
        digest, target = _index_location(spec, corpus_hash, embedding.revision)
        manifest_path = target / "index-manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"RankZephyr index missing: {manifest_path}; run --stage index or all")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("index_fingerprint") != digest:
            raise ValueError("RankZephyr index is incomplete or incompatible; rebuild it")
        store = DenseStore(target, f"{spec.dataset}_{digest[:16]}")
        try:
            with destination.open("w", encoding="utf-8") as handle:
                for number, question in enumerate(questions, 1):
                    started = time.perf_counter()
                    evidence, error = [], None
                    if question.available:
                        try:
                            vector = embedding.encode([question.question])[0]
                            evidence = store.search(question.language, vector, spec.candidates)
                        except Exception as exc:
                            if is_fatal_accelerator_error(exc):
                                raise
                            error = f"{type(exc).__name__}: {exc}"
                    row = {
                        "id": question.id, "question": question.question,
                        "candidates": [asdict(item) for item in evidence], "error": error,
                        "retrieval_latency_ms": (time.perf_counter() - started) * 1000,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    LOGGER.info("RankZephyr retrieval %s/%s: %s (%s candidates)",
                                number, len(questions), question.id, len(evidence))
            return manifest
        finally:
            store.close()
    finally:
        del embedding
        _release_gpu(device)


def _rerank(spec, candidates: Path, temporary: Path, output: Path) -> tuple[dict, dict]:
    config = spec.rank_zephyr
    requests = temporary / "requests.jsonl"
    results = temporary / "ranked.jsonl"
    diagnostics_path = output / "worker-diagnostics.json"
    pending = set()
    with candidates.open(encoding="utf-8") as inputs, requests.open("w", encoding="utf-8") as handle:
        for line in inputs:
            row = json.loads(line)
            if not row["candidates"]:
                continue
            pending.add(row["id"])
            request = {
                "id": row["id"], "question": row["question"],
                "candidates": [
                    {"id": item["chunk_id"], "text": item["text"], "score": item["rerank_score"]}
                    for item in row["candidates"]
                ],
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
    if not pending:
        diagnostics = {"status": "no_candidates"}
        write_json(diagnostics_path, diagnostics)
        return {}, diagnostics
    env = dict(os.environ)
    env.pop(spec.generator.api_key_env, None)
    env.pop("LLM_API_KEY", None)
    env["NO_PROXY"] = ",".join(filter(None, (env.get("NO_PROXY"), "127.0.0.1", "localhost")))
    env["no_proxy"] = env["NO_PROXY"]
    env["VLLM_NO_USAGE_STATS"] = "1"
    started = time.perf_counter()
    worker_error = None
    if diagnostics_path.exists():
        diagnostics_path.unlink()
    try:
        command = [
            config.worker_python, "-I", "-u", str(Path(__file__).with_name("worker.py")),
            "--parent-pid", str(os.getpid()), "--model", config.reranker_model,
            "--revision", config.reranker_revision, "--snapshot", config.reranker_snapshot,
            "--input", str(requests), "--output", str(results),
            "--diagnostics", str(diagnostics_path), "--server-log", str(output / "vllm.log"),
        ]
        if sys.platform != "linux":
            raise RuntimeError("RankZephyr reranking requires a Linux CUDA worker")
        if not Path(config.worker_python).is_file():
            raise FileNotFoundError(f"{config.worker_python}; run scripts/setup_rank_zephyr_ubuntu.sh")
        LOGGER.info("RankZephyr worker starting with %s", config.worker_python)
        subprocess.run(command, env=env, stdin=subprocess.DEVNULL, check=True)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        worker_error = f"{type(exc).__name__}: {exc}"

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.is_file() else {}
    if worker_error:
        diagnostics.update(status="failed", error=diagnostics.get("error") or worker_error)
        LOGGER.error("RankZephyr worker failed: %s", diagnostics["error"])
    by_id = {}
    if results.is_file():
        with results.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    diagnostics.update(status="failed", error="Incomplete RankZephyr worker output")
                    break
                qid = row["id"]
                if qid not in pending or qid in by_id:
                    diagnostics.update(status="failed", error=f"Unexpected or duplicate RankZephyr result: {qid}")
                    by_id[qid] = {"error": diagnostics["error"]}
                else:
                    by_id[qid] = row
    missing = pending - by_id.keys()
    if missing:
        diagnostics.update(status="failed", error=diagnostics.get("error") or "RankZephyr worker omitted results")
        for qid in missing:
            by_id[qid] = {"error": diagnostics["error"]}
    diagnostics["wall_time_ms"] = round((time.perf_counter() - started) * 1000, 3)
    write_json(diagnostics_path, diagnostics)
    return by_id, diagnostics


class _PreparedAnswer:
    def __init__(self, runtime, evidence: list[RetrievedEvidence], error: str | None):
        self.runtime = runtime
        self.evidence = evidence
        self.error = error

    def answer(self, _collection, question: str) -> TechniqueResult:
        if self.error:
            raise RuntimeError(self.error)
        if not self.evidence:
            return TechniqueResult("", "", [])
        context = "\n\n".join(item.text for item in self.evidence)
        prompt = (
            "Use the supplied context to answer the question. Answer in the same language "
            "as the question. If the context does not support an answer, leave it blank.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        started = time.perf_counter()
        prediction = self.runtime.generate_prompt(prompt).strip()
        return TechniqueResult(prediction, context, self.evidence, {
            "generation_latency_ms": (time.perf_counter() - started) * 1000,
        })


def _evaluate_variant(spec, questions, corpus_hash, candidates, reranked, variant, runtime, output, device) -> dict:
    destination = output / variant
    destination.mkdir(parents=True, exist_ok=True)
    model_fingerprint = spec.generator.fingerprint()
    decoding = spec.generator.decoding_dict()
    predictions = []
    with candidates.open(encoding="utf-8") as inputs, (destination / "predictions.jsonl").open(
        "w", encoding="utf-8",
    ) as handle:
        for number, question in enumerate(questions, 1):
            row = json.loads(next(inputs))
            if row["id"] != question.id:
                raise ValueError("RankZephyr candidate QA order changed")
            evidence = [RetrievedEvidence(**item) for item in row["candidates"]]
            ids = [item.chunk_id for item in evidence]
            selected = evidence[:spec.top_k]
            result = reranked.get(question.id, {}) if variant == VARIANTS[1] else {}
            error = row["error"] or result.get("error")
            if variant == VARIANTS[1] and ids and not error:
                ranked_ids = result.get("ranked_ids", [])
                if sorted(ranked_ids) != sorted(ids) or len(set(ranked_ids)) != len(ids):
                    error = "Invalid RankZephyr candidate permutation"
                else:
                    by_id = dict(zip(ids, evidence))
                    selected = [
                        replace(by_id[chunk_id], rerank_score=1.0 / rank)
                        for rank, chunk_id in enumerate(ranked_ids[:spec.top_k], 1)
                    ]
            if error:
                selected = []
            record, = evaluate_questions(
                [question], _PreparedAnswer(runtime, selected, error), spec.dataset,
                variant, model_fingerprint, corpus_hash, decoding,
            )
            if record.error and not error and is_fatal_accelerator_error(RuntimeError(record.error)):
                raise RuntimeError(f"{variant} generation failed for {question.id}: {record.error}")
            # Keep evidence metrics when the answer generator fails after retrieval.
            record.retrieved_source_ids = list(dict.fromkeys(item.source_id for item in selected))
            record.rerank_scores = [item.rerank_score for item in selected]
            if question.source_ids:
                relevant = _relevant_ranks(selected, question.source_ids)
                record.diagnostics.update(
                    relevant_ranks=relevant, source_recall_at_10=float(bool(relevant)),
                    candidate_relevant_ranks=_relevant_ranks(evidence, question.source_ids, spec.candidates),
                )
            retrieval_ms = row["retrieval_latency_ms"]
            rerank_ms = result.get("rerank_latency_ms", 0.0)
            record.latency_ms = round(record.latency_ms + retrieval_ms + rerank_ms, 3)
            record.diagnostics.update({
                "candidate_pool_hash": fingerprint(ids), "candidate_cutoff": spec.candidates,
                "retrieved_candidates": len(ids), "ranked_chunk_ids": [item.chunk_id for item in selected],
                "rerank_windows": result.get("windows", 0),
                "rerank_repaired_windows": result.get("repaired_windows", 0),
                "rerank_repairs": result.get("repairs", []),
                "retrieval_latency_ms": retrieval_ms, "rerank_latency_ms": rerank_ms,
                "retrieval_device": device,
                "gpu_fallback": device == "cpu" and os.getenv("RAG_DEVICE", "auto") != "cpu",
            })
            payload = record.to_dict()
            predictions.append(payload)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            LOGGER.info("%s generation %s/%s: %s (%s)", variant, number, len(questions), question.id, record.status)
        if next(inputs, None) is not None:
            raise ValueError("RankZephyr candidate file contains extra QA")
    metrics = summarize(predictions)
    write_json(destination / "metrics.json", metrics)
    write_jsonl(destination / "missing-context.jsonl", (
        {"id": qa.id, "language": qa.language, "source_ids": qa.source_ids}
        for qa in questions if not qa.available
    ))
    return metrics


def run_evaluate(spec, bundle, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    questions = bundle.questions_for_split(spec.evaluation_split)
    corpus_hash = _corpus_fingerprint(bundle.documents)
    device = resolve_torch_device("RankZephyr evaluation")
    with tempfile.TemporaryDirectory(prefix="rag-rank-zephyr-") as directory:
        temporary = Path(directory)
        candidates = temporary / "candidates.jsonl"
        index = _retrieve(spec, questions, corpus_hash, candidates, device)
        rerank_output = output / VARIANTS[1]
        rerank_output.mkdir(exist_ok=True)
        reranked, diagnostics = _rerank(spec, candidates, temporary, rerank_output)
        LOGGER.info("RankZephyr generation: reranker stopped; loading shared generator")
        runtime = spec.generator.runtime()
        try:
            reports = {}
            for variant in VARIANTS:
                metrics = _evaluate_variant(
                    spec, questions, corpus_hash, candidates, reranked, variant, runtime, output, device,
                )
                is_reranked = variant == VARIANTS[1]
                metadata = {
                    "technique": variant, "created_at": _now(), "split": spec.evaluation_split,
                    "scheduled_questions": len(questions), "corpus_fingerprint": corpus_hash,
                    "model_fingerprint": spec.generator.fingerprint(),
                    "decoding_config": spec.generator.decoding_dict(),
                    "index": index, "candidates": spec.candidates, "top_k": spec.top_k,
                    "score_type": "reciprocal_rank" if is_reranked else "dense_cosine",
                    "reranker": diagnostics if is_reranked else None,
                }
                write_json(output / variant / "manifest.json", metadata)
                reports[variant] = (
                    {**metrics, "status": "failed", "error": diagnostics.get("error")}
                    if is_reranked and diagnostics.get("status") == "failed" else metrics
                )
            return reports
        finally:
            del runtime
            _release_gpu(device)
