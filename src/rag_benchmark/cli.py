from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .metrics import read_predictions, summarize, write_json, write_jsonl
from .shared.config import PipelineConfig, collection_name
from .shared.datasets import load_dataset
from .shared.device import resolve_torch_device
from .shared.evaluation import evaluate_questions
from .shared.local_llm import HFModelConfig, LocalLLM
from .shared.retrieval import CrossEncoderReranker, E5Embedding, QdrantStore
from .techniques import TECHNIQUES, get_technique
from .techniques.dillema.pipeline import RAGPipeline
from .techniques.rrag.config import RRAGConfig
from .techniques.rrag.pipeline import RRAGTechnique


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _technique_root(index_root: Path, technique: str, dataset: str) -> Path:
    return index_root / technique / dataset


def _index_manifest_path(index_root: Path, technique: str, dataset: str) -> Path:
    return _technique_root(index_root, technique, dataset) / "index-manifest.json"


def _languages(dataset: str) -> list[str]:
    return ["id"] if dataset == "its" else ["en", "it"]


def _make_index_runner(technique: str, store, embedder, pipeline_config, rrag_config):
    if technique == "dillema":
        return RAGPipeline(store, embedder, reranker=None, config=pipeline_config)
    return RRAGTechnique(store, embedder, pipeline_config=pipeline_config, rrag_config=rrag_config)


def run_index(technique: str, dataset: str, source: Path, index_root: Path) -> dict:
    get_technique(technique)
    bundle = load_dataset(dataset, source)
    pipeline_config = PipelineConfig.from_env()
    rrag_config = RRAGConfig.from_env()
    device = resolve_torch_device("RAG indexing")
    embedder = E5Embedding(pipeline_config.embedding_model, device=device)
    target = _technique_root(index_root, technique, dataset)
    store = QdrantStore(target)
    runner = _make_index_runner(technique, store, embedder, pipeline_config, rrag_config)
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
        "training_questions": len(bundle.questions_for_split("train")),
        "evaluation_questions": len(bundle.questions_for_split("test")),
        "collections": indexed,
        "pipeline_config": asdict(pipeline_config),
        "rrag_config": asdict(rrag_config) if technique == "rrag" else None,
        "pipeline_fingerprint": pipeline_config.fingerprint(),
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


def run_train(
    technique: str,
    dataset: str,
    source: Path | None,
    index_root: Path,
    output: Path,
) -> dict:
    spec = get_technique(technique)
    if not spec.supports_training:
        raise ValueError(f"Technique '{technique}' does not train a model; run index and evaluate instead")
    _manifest, dataset_source = _load_index_manifest(index_root, technique, dataset, source)
    bundle = load_dataset(dataset, dataset_source)
    pipeline_config = PipelineConfig.from_env()
    rrag_config = RRAGConfig.from_env()
    device = resolve_torch_device("R²AG training")
    runtime = LocalLLM(HFModelConfig.from_env())
    target = _technique_root(index_root, technique, dataset)
    store = QdrantStore(target)
    results = {}
    try:
        pipeline = RRAGTechnique(
            store, E5Embedding(pipeline_config.embedding_model, device=device), runtime=runtime,
            pipeline_config=pipeline_config, rrag_config=rrag_config,
        )
        from .techniques.rrag.trainer import train_rrag_language

        for language in _languages(dataset):
            questions = [
                question for question in bundle.questions
                if question.split == "train" and question.language == language
            ]
            results[language] = train_rrag_language(
                pipeline, runtime, questions, collection_name(dataset, language), output, language,
            )
    finally:
        store.close()
    manifest = {
        "schema_version": 1,
        "created_at": _now(),
        "technique": technique,
        "dataset": dataset,
        "source": str(dataset_source.resolve()),
        "model_fingerprint": runtime.fingerprint,
        "languages": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "training-manifest.json", manifest)
    return manifest


def _checkpoint_file(root: Path, language: str) -> Path:
    if root.is_file():
        return root
    return root / language / "checkpoint.pt"


def run_evaluate(
    technique: str,
    dataset: str,
    index_root: Path,
    output: Path,
    source: Path | None = None,
    checkpoint: Path | None = None,
) -> dict:
    get_technique(technique)
    index_manifest, dataset_source = _load_index_manifest(index_root, technique, dataset, source)
    bundle = load_dataset(dataset, dataset_source)
    scheduled = bundle.questions_for_split("test")
    pipeline_config = PipelineConfig.from_env()
    rrag_config = RRAGConfig.from_env()
    device = resolve_torch_device("RAG evaluation")
    runtime = LocalLLM(HFModelConfig.from_env())
    store = QdrantStore(_technique_root(index_root, technique, dataset))
    prediction_records = []
    try:
        embedder = E5Embedding(pipeline_config.embedding_model, device=device)
        if technique == "dillema":
            runner = RAGPipeline(
                store, embedder, CrossEncoderReranker(pipeline_config.reranker_model, device=device),
                pipeline_config, generator=runtime,
            )
            prediction_records = evaluate_questions(
                scheduled, runner, dataset, technique, runtime.fingerprint,
            )
        else:
            if checkpoint is None:
                raise ValueError("--checkpoint is required for RRAG evaluation")
            from .techniques.rrag.r2former import R2AGAdapter

            for language in _languages(dataset):
                adapter, _payload = R2AGAdapter.load_checkpoint(
                    _checkpoint_file(checkpoint, language), runtime.fingerprint,
                    map_location=runtime.device,
                )
                runner = RRAGTechnique(
                    store, embedder, runtime=runtime, adapter=adapter,
                    pipeline_config=pipeline_config, rrag_config=rrag_config,
                )
                language_questions = [q for q in scheduled if q.language == language]
                prediction_records.extend(evaluate_questions(
                    language_questions, runner, dataset, technique, runtime.fingerprint,
                ))
    finally:
        store.close()

    output.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in prediction_records]
    if len(rows) != len(scheduled):
        raise RuntimeError("Evaluation did not produce exactly one prediction per scheduled test question")
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
        "rrag": asdict(rrag_config) if technique == "rrag" else None,
        "hf_model": asdict(runtime.config),
        "pipeline_fingerprint": pipeline_config.fingerprint(),
        "model_fingerprint": runtime.fingerprint,
    }
    write_json(output / "config.json", config_record)
    manifest = {
        "schema_version": 2,
        "created_at": _now(),
        "technique": technique,
        "dataset": dataset,
        "split": "test",
        "source": str(dataset_source.resolve()),
        "scheduled_questions": len(scheduled),
        "prediction_records": len(rows),
        "missing_context": len(missing),
        "model_fingerprint": runtime.fingerprint,
        "index_manifest": index_manifest,
        "python": platform.python_version(),
        "files": ["manifest.json", "predictions.jsonl", "bleu-summary.json", "missing-context.jsonl", "config.json"],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def run_compare(its_run: Path, uniqa_run: Path, output: Path) -> dict:
    its_summary = summarize(read_predictions(its_run))
    uniqa_summary = summarize(read_predictions(uniqa_run))
    if its_summary.get("dataset") != "its" or uniqa_summary.get("dataset") != "uniqa":
        raise ValueError("--its-run must contain ITS records and --uniqa-run UniQA records")
    comparison = {
        "created_at": _now(),
        "warning": "Scores across Indonesian, English, and Italian are reported separately; they do not establish that one corpus is intrinsically better.",
        "its": its_summary["its"],
        "uniqa": {"strict": uniqa_summary["strict"], "corpus_valid": uniqa_summary["corpus_valid"]},
        "inputs": {"its": str(its_run.resolve()), "uniqa": str(uniqa_run.resolve())},
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", comparison)
    return comparison


def _run_identity(records: list[dict]) -> tuple:
    return (
        {row.get("dataset") for row in records},
        {row.get("split") for row in records},
        {row.get("model_fingerprint") for row in records},
        tuple(sorted(str(row.get("id")) for row in records)),
    )


def _summary_groups(summary: dict) -> dict[str, dict]:
    if summary["dataset"] == "its":
        return {"id": summary["its"]}
    return {
        "en": summary["strict"]["en"],
        "it": summary["strict"]["it"],
    }


def run_compare_techniques(baseline_run: Path, candidate_run: Path, output: Path) -> dict:
    baseline = read_predictions(baseline_run)
    candidate = read_predictions(candidate_run)
    if _run_identity(baseline) != _run_identity(candidate):
        raise ValueError("Technique runs must have identical dataset, split, model fingerprint, and QA IDs")
    baseline_summary = summarize(baseline)
    candidate_summary = summarize(candidate)
    deltas = {}
    for language, baseline_group in _summary_groups(baseline_summary).items():
        candidate_group = _summary_groups(candidate_summary)[language]
        deltas[language] = {
            "BLEU_raw": candidate_group["BLEU_raw"]["score"] - baseline_group["BLEU_raw"]["score"],
            "BLEU_content": candidate_group["BLEU_content"]["score"] - baseline_group["BLEU_content"]["score"],
            "recall_at_10": (
                candidate_group["retrieval"]["recall_at_10"] - baseline_group["retrieval"]["recall_at_10"]
                if candidate_group["retrieval"]["recall_at_10"] is not None
                and baseline_group["retrieval"]["recall_at_10"] is not None else None
            ),
            "average_latency_ms": candidate_group["average_latency_ms"] - baseline_group["average_latency_ms"],
        }
    comparison = {
        "created_at": _now(),
        "baseline_technique": baseline[0].get("technique") if baseline else None,
        "candidate_technique": candidate[0].get("technique") if candidate else None,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "candidate_minus_baseline": deltas,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "technique-comparison.json", comparison)
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag_benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    techniques = tuple(TECHNIQUES)

    index = subparsers.add_parser("index", help="Build a technique-isolated RAG index")
    index.add_argument("--technique", choices=techniques, required=True)
    index.add_argument("--dataset", choices=("its", "uniqa"), required=True)
    index.add_argument("--source", type=Path, required=True)
    index.add_argument("--index-root", type=Path, default=Path("artifacts/indexes"))

    train = subparsers.add_parser("train", help="Train the RRAG auxiliary R²-Former")
    train.add_argument("--technique", choices=techniques, required=True)
    train.add_argument("--dataset", choices=("its", "uniqa"), required=True)
    train.add_argument("--source", type=Path)
    train.add_argument("--index-root", type=Path, default=Path("artifacts/indexes"))
    train.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="Run technique generation and BLEU")
    evaluate.add_argument("--technique", choices=techniques, required=True)
    evaluate.add_argument("--dataset", choices=("its", "uniqa"), required=True)
    evaluate.add_argument("--index-root", type=Path, default=Path("artifacts/indexes"))
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--source", type=Path)
    evaluate.add_argument("--checkpoint", type=Path)

    compare = subparsers.add_parser("compare", help="Compare completed ITS and UniQA runs")
    compare.add_argument("--its-run", type=Path, required=True)
    compare.add_argument("--uniqa-run", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    techniques_compare = subparsers.add_parser("compare-techniques", help="Compare controlled technique runs")
    techniques_compare.add_argument("--baseline-run", type=Path, required=True)
    techniques_compare.add_argument("--candidate-run", type=Path, required=True)
    techniques_compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.command == "index":
        result = run_index(args.technique, args.dataset, args.source, args.index_root)
    elif args.command == "train":
        result = run_train(args.technique, args.dataset, args.source, args.index_root, args.output)
    elif args.command == "evaluate":
        result = run_evaluate(
            args.technique, args.dataset, args.index_root, args.output, args.source, args.checkpoint,
        )
    elif args.command == "compare":
        result = run_compare(args.its_run, args.uniqa_run, args.output)
    else:
        result = run_compare_techniques(args.baseline_run, args.candidate_run, args.output)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")

