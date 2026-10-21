from __future__ import annotations

import json
import logging
import os
import platform
from dataclasses import asdict
from pathlib import Path

from ...shared.api_llm import APIModelConfig, OpenAICompatibleLLM
from ...shared.artifacts import _corpus_fingerprint, _index_manifest_path, _languages, _now, _technique_root
from ...shared.config import collection_name
from ...shared.datasets import load_dataset
from ...shared.device import resolve_torch_device
from ...shared.evaluation import evaluate_questions
from ...shared.local_llm import HFModelConfig, LocalLLM
from ...shared.metrics import summarize, write_json, write_jsonl
from ..registry import get_technique
from .config import PipelineConfig
from .embedding import E5Embedding
from .generation import DiLLeMaGenerator
from .pipeline import RAGPipeline
from .rerank import CrossEncoderReranker
from .store import QdrantStore

LOGGER = logging.getLogger(__name__)


def run_index(
    technique: str, dataset: str, source: Path, index_root: Path,
    pipeline_config: PipelineConfig | None = None,
) -> dict:
    get_technique(technique)
    bundle = load_dataset(dataset, source)
    pipeline_config = pipeline_config or PipelineConfig.from_env()
    device = resolve_torch_device("RAG indexing")
    embedder = E5Embedding(pipeline_config.embedding_model, device=device)
    target = _technique_root(index_root, technique, dataset)
    store = QdrantStore(target)
    runner = RAGPipeline(store, embedder, reranker=None, config=pipeline_config)
    indexed: dict[str, dict] = {}
    try:
        for language in _languages(dataset):
            documents = [document for document in bundle.documents if document.language == language]
            name = collection_name(dataset, language)
            chunks = runner.index(name, documents)
            indexed[name] = {
                "language": language,
                "documents": len(documents),
                "chunks": chunks,
                "points": store.count(name),
            }
    finally:
        store.close()
    manifest = {
        "schema_version": 2,
        "created_at": _now(),
        "technique": technique,
        "dataset": dataset,
        "source": str(source.resolve()),
        "qa_questions": {
            "all": len(bundle.questions_for_split("all")),
            "train_label": len(bundle.questions_for_split("train")),
            "test_label": len(bundle.questions_for_split("test")),
        },
        "collections": indexed,
        "pipeline_config": asdict(pipeline_config),
        "pipeline_fingerprint": pipeline_config.fingerprint(),
        "corpus_fingerprint": _corpus_fingerprint(bundle.documents),
        "indexing_is_model_training": False,
    }
    path = _index_manifest_path(index_root, technique, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    return manifest


def _load_index_manifest(index_root: Path, technique: str, dataset: str, source: Path | None):
    path = _index_manifest_path(index_root, technique, dataset)
    if not path.is_file() and source is None:
        raise FileNotFoundError(f"Index manifest not found: {path}. Run index first or pass --source.")
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    dataset_source = source or Path(manifest["source"])
    return manifest, Path(dataset_source)


def _evaluation_runtime():
    if APIModelConfig.is_configured():
        config = APIModelConfig.from_env()
        LOGGER.info("Using OpenAI-compatible generator %s at %s", config.model, config.base_url)
        return OpenAICompatibleLLM(config)
    LOGGER.info("Using local Hugging Face generator")
    return LocalLLM(HFModelConfig.from_env())


def _decoding_config(runtime) -> dict:
    if isinstance(runtime, OpenAICompatibleLLM):
        return {"temperature": 0.0, "max_tokens": runtime.config.max_tokens}
    return {"temperature": 0.0, "max_tokens": runtime.config.max_new_tokens, "do_sample": False}


def run_evaluate(
    technique: str,
    dataset: str,
    index_root: Path,
    output: Path,
    source: Path | None = None,
    runtime=None,
    model_fingerprint: str | None = None,
    decoding_config: dict | None = None,
    evaluation_split: str = "test",
    pipeline_config: PipelineConfig | None = None,
) -> dict:
    get_technique(technique)
    index_manifest, dataset_source = _load_index_manifest(index_root, technique, dataset, source)
    bundle = load_dataset(dataset, dataset_source)
    scheduled = bundle.questions_for_split(evaluation_split)
    pipeline_config = pipeline_config or PipelineConfig.from_env()
    device = resolve_torch_device("RAG evaluation")
    runtime = runtime or _evaluation_runtime()
    resolved_model_fingerprint = model_fingerprint or runtime.fingerprint
    resolved_decoding = decoding_config or _decoding_config(runtime)
    corpus_fingerprint = (index_manifest or {}).get("corpus_fingerprint") or _corpus_fingerprint(bundle.documents)
    store = QdrantStore(_technique_root(index_root, technique, dataset))
    prediction_records = []
    try:
        embedder = E5Embedding(pipeline_config.embedding_model, device=device)
        runner = RAGPipeline(
            store, embedder, CrossEncoderReranker(pipeline_config.reranker_model, device=device),
            pipeline_config, generator=DiLLeMaGenerator(runtime),
        )
        prediction_records = evaluate_questions(
            scheduled, runner, dataset, technique, resolved_model_fingerprint,
            corpus_fingerprint, resolved_decoding,
        )
        gpu_fallback = device == "cpu" and os.getenv("RAG_DEVICE", "auto").casefold() != "cpu"
        for record in prediction_records:
            record.diagnostics["gpu_fallback"] = gpu_fallback
            record.diagnostics["device"] = device
    finally:
        store.close()

    output.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in prediction_records]
    if len(rows) != len(scheduled):
        raise RuntimeError("Evaluation did not produce exactly one prediction per scheduled question")
    write_jsonl(output / "predictions.jsonl", rows)
    missing = [question for question in scheduled if not question.available]
    write_jsonl(output / "missing-context.jsonl", [{
        "id": item.id, "dataset": item.dataset, "language": item.language,
        "question": item.question, "source_ids": item.source_ids,
    } for item in missing])
    summary = summarize(rows)
    write_json(output / "bleu-summary.json", summary)
    config_record = {
        "technique": technique,
        "pipeline": asdict(pipeline_config),
        "generator": {
            "kind": runtime.kind,
            "config": (
                runtime.config.public_dict()
                if isinstance(runtime, OpenAICompatibleLLM)
                else asdict(runtime.config)
            ),
        },
        "pipeline_fingerprint": pipeline_config.fingerprint(),
        "model_fingerprint": resolved_model_fingerprint,
        "corpus_fingerprint": corpus_fingerprint,
        "decoding_config": resolved_decoding,
        "evaluation_split": evaluation_split,
    }
    write_json(output / "config.json", config_record)
    manifest = {
        "schema_version": 2,
        "created_at": _now(),
        "technique": technique,
        "dataset": dataset,
        "split": evaluation_split,
        "source": str(dataset_source.resolve()),
        "scheduled_questions": len(scheduled),
        "prediction_records": len(rows),
        "missing_context": len(missing),
        "model_fingerprint": resolved_model_fingerprint,
        "corpus_fingerprint": corpus_fingerprint,
        "decoding_config": resolved_decoding,
        "index_manifest": index_manifest,
        "python": platform.python_version(),
        "files": ["manifest.json", "predictions.jsonl", "bleu-summary.json", "missing-context.jsonl", "config.json"],
    }
    write_json(output / "manifest.json", manifest)
    return manifest
