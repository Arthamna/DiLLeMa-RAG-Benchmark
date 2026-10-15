from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import LLMConfig, PipelineConfig, collection_name
from .datasets import DatasetBundle, load_dataset
from .embedding import E5Embedding
from .generation import OpenAIGenerator
from .metrics import read_predictions, summarize, write_json, write_jsonl
from .models import PredictionRecord, QARecord
from .pipeline import RAGPipeline
from .rerank import CrossEncoderReranker
from .store import QdrantStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_fingerprint(config: PipelineConfig, llm_model: str | None = None) -> str:
    value = {
        "embedding": config.embedding_model,
        "reranker": config.reranker_model,
        "generator": llm_model,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _index_manifest_path(index_root: Path, dataset: str) -> Path:
    return index_root / "manifests" / f"{dataset}-index.json"


def run_index(dataset: str, source: Path, index_root: Path) -> dict:
    bundle = load_dataset(dataset, source)
    config = PipelineConfig.from_env()
    embedder = E5Embedding(config.embedding_model)
    store = QdrantStore(index_root)
    pipeline = RAGPipeline(store, embedder, reranker=None, config=config)
    indexed: dict[str, dict] = {}
    try:
        languages = ["id"] if dataset == "its" else ["en", "it"]
        for language in languages:
            documents = [document for document in bundle.documents if document.language == language]
            name = collection_name(dataset, language)
            chunks = pipeline.index(name, documents)
            indexed[name] = {
                "language": language,
                "documents": len(documents),
                "chunks": chunks,
                "points": store.count(name),
            }
    finally:
        store.close()

    manifest = {
        "schema_version": 1,
        "created_at": _now(),
        "dataset": dataset,
        "source": str(source.resolve()),
        "question_count": len(bundle.questions),
        "missing_context_count": len(bundle.missing_questions),
        "collections": indexed,
        "pipeline_config": asdict(config),
        "pipeline_fingerprint": config.fingerprint(),
        "model_fingerprint": _model_fingerprint(config),
        "indexing_is_fine_tuning": False,
    }
    path = _index_manifest_path(index_root, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    return manifest


def evaluate_questions(
    questions: list[QARecord],
    pipeline,
    generator,
    dataset: str,
) -> list[PredictionRecord]:
    predictions: list[PredictionRecord] = []
    for question in questions:
        started = time.perf_counter()
        prediction = ""
        evidence = []
        context = ""
        error = None
        if not question.available:
            status = "missing_context"
        else:
            try:
                packed = pipeline.retrieve(
                    collection_name(dataset, question.language),
                    question.question,
                )
                evidence = packed.evidence
                context = packed.text
                if not context:
                    status = "no_evidence"
                else:
                    prediction = generator.generate(question.question, context)
                    status = "ok" if prediction else "abstention"
            except Exception as exc:  # inference failures stay in the BLEU denominator
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                prediction = ""
        predictions.append(
            PredictionRecord(
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
            )
        )
    return predictions


def run_evaluate(dataset: str, index_root: Path, output: Path, source: Path | None = None) -> dict:
    index_manifest_path = _index_manifest_path(index_root, dataset)
    if not index_manifest_path.is_file() and source is None:
        raise FileNotFoundError(
            f"Index manifest not found: {index_manifest_path}. Run index first or pass --source."
        )
    index_manifest = (
        json.loads(index_manifest_path.read_text(encoding="utf-8"))
        if index_manifest_path.is_file()
        else None
    )
    dataset_source = source or Path(index_manifest["source"])
    bundle = load_dataset(dataset, dataset_source)
    config = PipelineConfig.from_env()
    llm = LLMConfig.from_env()
    store = QdrantStore(index_root)
    try:
        pipeline = RAGPipeline(
            store,
            E5Embedding(config.embedding_model),
            CrossEncoderReranker(config.reranker_model),
            config,
        )
        generator = OpenAIGenerator(llm, config)
        prediction_records = evaluate_questions(bundle.questions, pipeline, generator, dataset)
    finally:
        store.close()

    output.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in prediction_records]
    if len(rows) != len(bundle.questions):
        raise RuntimeError("Evaluation did not produce exactly one prediction per scheduled question")
    write_jsonl(output / "predictions.jsonl", rows)
    write_jsonl(
        output / "missing-context.jsonl",
        [
            {
                "id": item.id,
                "dataset": item.dataset,
                "language": item.language,
                "question": item.question,
                "source_ids": item.source_ids,
            }
            for item in bundle.missing_questions
        ],
    )
    summary = summarize(rows)
    write_json(output / "bleu-summary.json", summary)
    config_record = {
        "pipeline": asdict(config),
        "llm": {"base_url": llm.base_url, "model": llm.model},
        "pipeline_fingerprint": config.fingerprint(),
        "model_fingerprint": _model_fingerprint(config, llm.model),
    }
    write_json(output / "config.json", config_record)
    manifest = {
        "schema_version": 1,
        "created_at": _now(),
        "dataset": dataset,
        "source": str(Path(dataset_source).resolve()),
        "scheduled_questions": len(bundle.questions),
        "prediction_records": len(rows),
        "missing_context": len(bundle.missing_questions),
        "python": platform.python_version(),
        "files": [
            "manifest.json",
            "predictions.jsonl",
            "bleu-summary.json",
            "missing-context.jsonl",
            "config.json",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def run_compare(its_run: Path, uniqa_run: Path, output: Path) -> dict:
    its_records = read_predictions(its_run)
    uniqa_records = read_predictions(uniqa_run)
    its_summary = summarize(its_records)
    uniqa_summary = summarize(uniqa_records)
    if its_summary.get("dataset") != "its" or uniqa_summary.get("dataset") != "uniqa":
        raise ValueError("--its-run must contain ITS records and --uniqa-run UniQA records")
    comparison = {
        "created_at": _now(),
        "warning": "Scores across Indonesian, English, and Italian are reported separately; they do not establish that one corpus is intrinsically better.",
        "its": its_summary["its"],
        "uniqa": {
            "strict": uniqa_summary["strict"],
            "corpus_valid": uniqa_summary["corpus_valid"],
        },
        "inputs": {"its": str(its_run.resolve()), "uniqa": str(uniqa_run.resolve())},
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", comparison)
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag_benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Build a RAG index (not model fine-tuning)")
    index.add_argument("--dataset", choices=("its", "uniqa"), required=True)
    index.add_argument("--source", type=Path, required=True)
    index.add_argument("--index-root", type=Path, default=Path("artifacts/indexes"))

    evaluate = subparsers.add_parser("evaluate", help="Run retrieval, generation, and BLEU")
    evaluate.add_argument("--dataset", choices=("its", "uniqa"), required=True)
    evaluate.add_argument("--index-root", type=Path, default=Path("artifacts/indexes"))
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--source", type=Path, help="Override source recorded by index")

    compare = subparsers.add_parser("compare", help="Compare completed ITS and UniQA runs")
    compare.add_argument("--its-run", type=Path, required=True)
    compare.add_argument("--uniqa-run", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "index":
        result = run_index(args.dataset, args.source, args.index_root)
    elif args.command == "evaluate":
        result = run_evaluate(args.dataset, args.index_root, args.output, args.source)
    else:
        result = run_compare(args.its_run, args.uniqa_run, args.output)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")

