from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ...shared.artifacts import _corpus_fingerprint, _languages, _now
from ...shared.config import collection_name
from ...shared.evaluation import evaluate_questions
from ...shared.metrics import summarize, write_json, write_jsonl
from .client import InferenceServiceClient, RetrievalServiceClient
from .indexing import deterministic_holdout, document_payload
from .pipeline import RetroPipeline
from .reporting import write_pareto

UPSTREAM = {"retro": "6945cdd", "piperag": "b6eb158"}


def _root(spec) -> Path:
    return spec.index_root / "retro" / spec.dataset


def _service_root(spec) -> str:
    return f"{spec.services.retrieval_index_root.rstrip('/')}/retro/{spec.dataset}"


def _manifest_path(spec) -> Path:
    return _root(spec) / "index-manifest.json"


def _index_id(spec, track: str, language: str, corpus_fingerprint: str) -> str:
    return f"{spec.dataset}:{language}:{track}:{corpus_fingerprint[:16]}:{spec.retro.fingerprint()[:12]}"


def run_index(spec, bundle, client: RetrievalServiceClient | None = None) -> dict[str, Any]:
    # Create the shared mount parent as the host user before the Docker retrieval
    # service writes its hashed index directories beneath it.
    path = _manifest_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    client = client or RetrievalServiceClient(spec.services)
    corpus_fingerprint = _corpus_fingerprint(bundle.documents)
    paper_train, paper_holdout = deterministic_holdout(
        bundle.documents, spec.paper.holdout_fraction, spec.paper.seed,
    )
    collections: dict[str, Any] = {}
    paper_collections: dict[str, Any] = {}
    try:
        for language in _languages(spec.dataset):
            documents = [item for item in bundle.documents if item.language == language]
            name = collection_name(spec.dataset, language)
            index_id = _index_id(spec, "qa", language, corpus_fingerprint)
            response = client.build_index({
                "index_id": index_id,
                "storage_root": _service_root(spec),
                "track": "qa",
                "documents": [document_payload(item) for item in documents],
                "tokenizer": spec.retro.tokenizer,
                "retriever_model": spec.retro.retriever_model,
                "chunk_size": spec.retro.chunk_size,
            })
            collections[name] = {"index_id": index_id, **response.get("index", {})}
            if spec.paper.enabled:
                train = [item for item in paper_train if item.language == language]
                holdout = [item for item in paper_holdout if item.language == language]
                paper_id = _index_id(spec, "paper", language, corpus_fingerprint)
                response = client.build_index({
                    "index_id": paper_id,
                    "storage_root": _service_root(spec),
                    "track": "paper",
                    "documents": [document_payload(item) for item in train],
                    "tokenizer": spec.retro.tokenizer,
                    "retriever_model": spec.retro.retriever_model,
                    "chunk_size": spec.retro.chunk_size,
                })
                paper_collections[name] = {
                    "index_id": paper_id,
                    "holdout_ids": [item.id for item in holdout],
                    **response.get("index", {}),
                }
    finally:
        if owns_client:
            client.close()
    manifest = {
        "schema_version": 3,
        "created_at": _now(),
        "dataset": spec.dataset,
        "corpus_fingerprint": corpus_fingerprint,
        "retro_fingerprint": spec.retro.fingerprint(),
        "upstream": UPSTREAM,
        "collections": collections,
        "paper_collections": paper_collections,
        "paper_split": {
            "seed": spec.paper.seed,
            "holdout_fraction": spec.paper.holdout_fraction,
            "train_documents": len(paper_train),
            "holdout_documents": len(paper_holdout),
        },
    }
    write_json(path, manifest)
    return manifest


def _load_manifest(spec) -> dict[str, Any]:
    path = _manifest_path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"RETRO index manifest not found: {path}. Run --stage index first.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("retro_fingerprint") != spec.retro.fingerprint():
        raise ValueError("RETRO index was built with a different runtime configuration")
    return value


def _configure_payload(spec) -> dict[str, Any]:
    return {
        "checkpoint": str(spec.retro.checkpoint),
        "model_config": str(spec.retro.model_config),
        "onnx_encoder": str(spec.retro.onnx_encoder),
        "onnx_decoder": str(spec.retro.onnx_decoder),
        "tokenizer": spec.retro.tokenizer,
        "retrieval_endpoint": spec.services.retrieval_endpoint,
        "service_timeout_seconds": spec.services.timeout_seconds,
        "max_message_mb": spec.services.max_message_mb,
        "retro": spec.retro.public_dict(),
        "piperag": spec.piperag.public_dict() if spec.piperag else {},
    }


def _write_technique_output(output: Path, rows: list[dict], config: dict[str, Any]) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions.jsonl", rows)
    summary = summarize(rows)
    write_json(output / "metrics.json", summary)
    write_json(output / "config.json", config)
    write_json(output / "manifest.json", {
        "schema_version": 3,
        "created_at": _now(),
        "technique": config["technique"],
        "dataset": config["dataset"],
        "prediction_records": len(rows),
        "python": platform.python_version(),
        "upstream": UPSTREAM,
        "runtime": config.get("runtime"),
        "profiles": config.get("profiles"),
        "files": ["manifest.json", "predictions.jsonl", "metrics.json", "config.json"],
    })
    return summary


def run_evaluate(spec, bundle, output: Path, variants: tuple[str, ...]) -> dict[str, dict]:
    from ..piperag.pipeline import PipeRagPipeline

    manifest = _load_manifest(spec)
    retrieval_client = RetrievalServiceClient(spec.services)
    for value in (*manifest["collections"].values(), *manifest.get("paper_collections", {}).values()):
        retrieval_client.load_index({
            "index_id": value["index_id"],
            "storage_root": _service_root(spec),
            "tokenizer": spec.retro.tokenizer,
            "retriever_model": spec.retro.retriever_model,
        })
    retrieval_client.close()
    client = InferenceServiceClient(spec.services)
    configured = client.configure(_configure_payload(spec))
    model_fingerprint = str(configured.get("model_fingerprint") or spec.retro.fingerprint())
    indexes = {name: value["index_id"] for name, value in manifest["collections"].items()}
    questions = bundle.questions_for_split(spec.evaluation_split)
    corpus_fingerprint = manifest["corpus_fingerprint"]
    reports: dict[str, dict] = {}
    profiles: dict[str, Any] = {}
    try:
        if "piperag" in variants:
            for collection, index_id in indexes.items():
                language = "id" if collection == "its" else collection.rsplit("_", 1)[-1]
                question = next((item.question for item in questions if item.language == language), "Retrieval augmented generation")
                profiles[collection] = client.profile_generation({
                    "index_id": index_id, "question": question,
                    "warmups": spec.paper.latency_warmups,
                    "repetitions": spec.paper.latency_repetitions,
                    "max_new_tokens": spec.retro.max_new_tokens,
                    "nprobe": spec.retro.nprobe,
                }).get("profile", {})
            write_json(output / "generation-profile.json", profiles)
        for variant in variants:
            generation = {
                "chunk_size": spec.retro.chunk_size,
                "num_neighbours": spec.retro.num_neighbours,
                "continuation_chunks": spec.retro.continuation_chunks,
                "retrieval_interval": spec.retro.retrieval_interval,
                "stale_steps": spec.piperag.stale_steps if variant == "piperag" else 0,
                "nprobe": spec.retro.nprobe,
                "dynamic_nprobe": False,
                "max_new_tokens": spec.retro.max_new_tokens,
            }
            pipeline_cls = PipeRagPipeline if variant == "piperag" else RetroPipeline
            pipeline = pipeline_cls(client, indexes, generation)
            records = evaluate_questions(
                questions, pipeline, spec.dataset, variant, model_fingerprint,
                corpus_fingerprint, {"do_sample": False, "max_new_tokens": spec.retro.max_new_tokens},
            )
            rows = [record.to_dict() for record in records]
            reports[variant] = _write_technique_output(output / variant, rows, {
                "technique": variant,
                "dataset": spec.dataset,
                "model_fingerprint": model_fingerprint,
                "corpus_fingerprint": corpus_fingerprint,
                "generation": generation,
                "retro": spec.retro.public_dict(),
                "piperag": spec.piperag.public_dict() if variant == "piperag" else None,
                "runtime": configured,
                "profiles": profiles,
            })
    finally:
        client.close()
    return reports


def run_paper_evaluation(spec, bundle, output: Path, variants: tuple[str, ...]) -> list[dict[str, Any]]:
    if not spec.paper.enabled or not variants:
        return []
    manifest = _load_manifest(spec)
    _, holdout = deterministic_holdout(bundle.documents, spec.paper.holdout_fraction, spec.paper.seed)
    client = InferenceServiceClient(spec.services)
    retrieval = RetrievalServiceClient(spec.services)
    try:
        for value in manifest["paper_collections"].values():
            retrieval.load_index({
                "index_id": value["index_id"], "storage_root": _service_root(spec),
                "tokenizer": spec.retro.tokenizer, "retriever_model": spec.retro.retriever_model,
            })
        client.configure(_configure_payload(spec))
        response = client.paper_evaluate({
            "indexes": {name: value["index_id"] for name, value in manifest["paper_collections"].items()},
            "nlists": {name: int(value["nlist"]) for name, value in manifest["paper_collections"].items()},
            "documents": [document_payload(item) for item in holdout],
            "variants": list(variants),
            "retro": spec.retro.public_dict(),
            "piperag": spec.piperag.public_dict() if spec.piperag else {},
            "paper": asdict(spec.paper),
        })
    finally:
        retrieval.close()
        client.close()
    rows = list(response.get("rows") or [])
    write_pareto(output / "paper", rows)
    write_json(output / "paper" / "profile.json", response.get("profile") or {})
    return rows


def doctor(spec) -> dict[str, Any]:
    retrieval = RetrievalServiceClient(spec.services)
    inference = InferenceServiceClient(spec.services)
    index_validation: dict[str, Any] = {"status": "not_built"}
    try:
        if _manifest_path(spec).is_file():
            manifest = _load_manifest(spec)
            loaded = {}
            for value in (*manifest["collections"].values(), *manifest.get("paper_collections", {}).values()):
                response = retrieval.load_index({
                    "index_id": value["index_id"], "storage_root": _service_root(spec),
                    "tokenizer": spec.retro.tokenizer, "retriever_model": spec.retro.retriever_model,
                })
                loaded[value["index_id"]] = response.get("index")
            index_validation = {"status": "ok", "loaded": loaded}
        retrieval_status = retrieval.health()
        configured = inference.configure(_configure_payload(spec))
        inference_status = inference.health()
    finally:
        retrieval.close()
        inference.close()
    environment = configured.get("environment") or {}
    current_hardware = hashlib.sha256(json.dumps({
        "device": environment.get("device"),
        "onnx_providers": environment.get("onnx_providers"),
        "onnxruntime": environment.get("onnxruntime"),
        "model": configured.get("model_fingerprint"),
    }, sort_keys=True).encode()).hexdigest()
    profile_validation = {}
    for index_id, value in (retrieval_status.get("indexes") or {}).items():
        if not value.get("profile_hardware_fingerprint"):
            state, reason = "rebuild_required", "profile cache is missing"
        elif value.get("profile_hardware_fingerprint") != current_hardware:
            state, reason = "rebuild_required", "profile hardware fingerprint is stale"
        elif value.get("profile_model_fingerprint") != configured.get("model_fingerprint"):
            state, reason = "rebuild_required", "profile model fingerprint is stale"
        else:
            state, reason = "ok", "profile cache matches current model and hardware"
        profile_validation[index_id] = {"status": state, "reason": reason}
    return {
        "status": "ok",
        "protocol_version": spec.services.protocol_version,
        "retrieval": retrieval_status,
        "inference": inference_status,
        "configuration": configured,
        "index_validation": index_validation,
        "profile_validation": profile_validation,
        "assets": {
            "checkpoint": str(spec.retro.checkpoint),
            "model_config": str(spec.retro.model_config),
            "onnx_encoder": str(spec.retro.onnx_encoder),
            "onnx_decoder": str(spec.retro.onnx_decoder),
        },
    }
