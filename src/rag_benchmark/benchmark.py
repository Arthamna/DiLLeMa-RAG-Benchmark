from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .cli import _corpus_fingerprint, _languages, _now, run_evaluate, run_index
from .metrics import read_predictions, summarize, write_json, write_jsonl
from .shared.api_llm import APIModelConfig, OpenAICompatibleLLM
from .shared.config import PipelineConfig
from .shared.datasets import load_dataset
from .shared.evaluation import _relevant_ranks
from .shared.local_llm import HFModelConfig, LocalLLM
from .shared.records import PredictionRecord, RetrievedEvidence
from .shared.retrieval import DocumentChunker

LOGGER = logging.getLogger(__name__)

RANKIFY_PRESETS = (
    "rankify_bm25_basic",
    "rankify_bge_basic",
    "rankify_bm25_bge_rerank_basic",
    "rankify_bge_bge_rerank_basic",
    "rankify_hybrid_bge_rerank_basic",
    "rankify_hybrid_bge_rerank_cot",
    "rankify_hybrid_bge_rerank_incontext",
)


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    backend: str
    model: str
    temperature: float
    max_tokens: int
    base_url: str = ""
    api_key_env: str = "LLM_API_KEY"
    quantization: str = "4bit"
    trust_remote_code: bool = False
    max_input_tokens: int = 4096

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationSpec":
        backend = str(value.get("backend") or "").strip().casefold()
        if backend not in {"huggingface", "openai-compatible"}:
            raise ValueError("generator.backend must be huggingface or openai-compatible")
        model = str(value.get("model") or "").strip()
        model_env = str(value.get("model_env") or "").strip()
        if not model and model_env:
            model = os.getenv(model_env, "").strip()
        if not model:
            raise ValueError("Generator model is empty; set generator.model or its model_env")
        base_url = str(value.get("base_url") or "").strip()
        base_url_env = str(value.get("base_url_env") or "LLM_BASE_URL").strip()
        if not base_url and base_url_env:
            base_url = os.getenv(base_url_env, "").strip()
        api_key_env = str(value.get("api_key_env") or "LLM_API_KEY").strip()
        temperature = float(value.get("temperature", 0.0))
        if temperature != 0.0:
            raise ValueError("Controlled benchmark requires generator.temperature=0")
        max_tokens_env = str(value.get("max_tokens_env") or "").strip()
        max_input_tokens_env = str(value.get("max_input_tokens_env") or "").strip()
        return cls(
            backend=backend,
            model=model,
            temperature=temperature,
            max_tokens=int(os.getenv(max_tokens_env, str(value.get("max_tokens", 768)))) if max_tokens_env else int(value.get("max_tokens", 768)),
            base_url=base_url,
            api_key_env=api_key_env,
            quantization=str(value.get("quantization") or os.getenv("HF_QUANTIZATION", "4bit")),
            trust_remote_code=bool(value.get("trust_remote_code", False)),
            max_input_tokens=(
                int(os.getenv(max_input_tokens_env, str(value.get("max_input_tokens", 4096))))
                if max_input_tokens_env else int(value.get("max_input_tokens", 4096))
            ),
        )

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def decoding_dict(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens, "do_sample": False}

    def runtime(self):
        if self.backend == "openai-compatible":
            if not self.base_url:
                raise ValueError("OpenAI-compatible generation requires base_url or base_url_env")
            if not os.getenv(self.api_key_env, "").strip():
                raise ValueError(
                    f"OpenAI-compatible generation requires environment variable {self.api_key_env}"
                )
            return OpenAICompatibleLLM(APIModelConfig(
                base_url=self.base_url,
                api_key=os.environ[self.api_key_env],
                model=self.model,
                model_source=self.model,
                max_tokens=self.max_tokens,
            ))
        return LocalLLM(HFModelConfig(
            model_name=self.model,
            quantization=self.quantization,
            trust_remote_code=self.trust_remote_code,
            max_input_tokens=self.max_input_tokens,
            max_new_tokens=self.max_tokens,
        ))


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    dataset: str
    source: Path
    index_root: Path
    rankify_python: Path
    dense_model: str
    reranker_model: str
    candidates: int
    top_k: int
    rrf_k: int
    presets: tuple[str, ...]
    generator: GenerationSpec

    @classmethod
    def load(cls, path: Path) -> "ExperimentSpec":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        dataset = str(raw.get("dataset") or "").casefold()
        if dataset not in {"its", "uniqa"}:
            raise ValueError("dataset must be its or uniqa")
        base = path.resolve().parent

        def resolved(
            name: str,
            default: str | None = None,
            *,
            follow_symlinks: bool = True,
        ) -> Path:
            value = raw.get(name, default)
            if not value:
                raise ValueError(f"Missing required configuration: {name}")

            candidate = Path(str(value)).expanduser()
            path = candidate if candidate.is_absolute() else base / candidate

            if follow_symlinks:
                return path.resolve()

            # Jadikan absolut tanpa mengikuti symlink virtualenv.
            return Path(os.path.abspath(path))

        models = raw.get("models") or {}
        retrieval = raw.get("retrieval") or {}
        presets = tuple(raw.get("presets") or RANKIFY_PRESETS)
        if presets != RANKIFY_PRESETS:
            raise ValueError(f"presets must contain the controlled seven in order: {', '.join(RANKIFY_PRESETS)}")
        return cls(
            dataset=dataset,
            source=resolved("source"),
            index_root=resolved("index_root", "../../artifacts/indexes"),
            rankify_python=resolved(
                "rankify_python",
                "../../.venv-rankify/bin/python",
                follow_symlinks=False,
            ),
            dense_model=str(models.get("dense_multilingual") or "").strip(),
            reranker_model=str(models.get("reranker_multilingual") or "").strip(),
            candidates=int(retrieval.get("candidates", 40)),
            top_k=int(retrieval.get("top_k", 10)),
            rrf_k=int(retrieval.get("rrf_k", 60)),
            presets=presets,
            generator=GenerationSpec.from_dict(raw.get("generator") or {}),
        ).validate()

    def validate(self) -> "ExperimentSpec":
        if not self.source.exists():
            raise FileNotFoundError(f"Dataset source not found: {self.source}")
        if not self.dense_model or not self.reranker_model:
            raise ValueError("models.dense_multilingual and models.reranker_multilingual are required")
        if min(self.candidates, self.top_k, self.rrf_k) <= 0 or self.top_k > self.candidates:
            raise ValueError("retrieval values must be positive and top_k cannot exceed candidates")
        return self

    def public_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source": str(self.source),
            "index_root": str(self.index_root),
            "rankify_python": str(self.rankify_python),
            "models": {"dense_multilingual": self.dense_model, "reranker_multilingual": self.reranker_model},
            "retrieval": {"candidates": self.candidates, "top_k": self.top_k, "rrf_k": self.rrf_k},
            "presets": list(self.presets),
            "generator": self.generator.public_dict(),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def index_fingerprint(self) -> str:
        pipeline = asdict(PipelineConfig.from_env())
        payload = {
            "dataset": self.dataset,
            "source": str(self.source),
            "models": {"dense": self.dense_model, "reranker": self.reranker_model},
            "retrieval": {"candidates": self.candidates, "top_k": self.top_k, "rrf_k": self.rrf_k},
            "chunking": {
                key: pipeline[key] for key in (
                    "chunk_size", "chunk_overlap", "min_chunk_chars", "parent_window_chars",
                    "embedding_model", "reranker_model",
                )
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class RankifyWorkerClient:
    def __init__(self, python: Path):
        self.python = python

    def invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.python.is_file():
            raise FileNotFoundError(
                f"Rankify worker Python not found: {self.python}. Run scripts/setup_rankify_ubuntu.sh"
            )
        worker_env = os.environ.copy()
        worker_env["PATH"] = os.pathsep.join([
            str(self.python.parent),
            worker_env.get("PATH", ""),
        ])

        completed = subprocess.run(
            [str(self.python), "-m", "rag_benchmark_rankify.worker"],
            input=json.dumps(request, ensure_ascii=False) + "\n",
            text=True,
            capture_output=True,
            check=False,
            env=worker_env,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Rankify worker failed ({completed.returncode}): {completed.stderr.strip() or completed.stdout.strip()}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("Rankify worker returned no JSON response")
        response = json.loads(lines[-1])
        if response.get("status") != "ok":
            raise RuntimeError(f"Rankify worker error: {response.get('error', 'unknown error')}")
        return response


def _rankify_index_root(spec: ExperimentSpec) -> Path:
    return spec.index_root / spec.index_fingerprint()[:16] / "rankify" / spec.dataset


def _dillema_index_root(spec: ExperimentSpec) -> Path:
    return spec.index_root / spec.index_fingerprint()[:16]


def _export_corpora(spec: ExperimentSpec, bundle) -> dict[str, Path]:
    config = PipelineConfig.from_env()
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


def _worker_common(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "dense_model": spec.dense_model,
        "reranker_model": spec.reranker_model,
        "candidates": spec.candidates,
        "top_k": spec.top_k,
        "rrf_k": spec.rrf_k,
        "generator": spec.generator.public_dict(),
        "device": os.getenv("RAG_DEVICE", "auto"),
    }


def _index_rankify(spec: ExperimentSpec, bundle, worker: RankifyWorkerClient) -> dict[str, Any]:
    corpora = _export_corpora(spec, bundle)
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


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _convert_worker_rows(
    rows: list[dict], questions: list, technique: str, spec: ExperimentSpec, corpus_fingerprint: str,
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
    spec: ExperimentSpec, bundle, worker: RankifyWorkerClient, output: Path, corpus_fingerprint: str,
) -> dict[str, Any]:
    questions = bundle.questions_for_split("test")
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


def _leaderboard_rows(reports: dict[str, dict]) -> list[dict[str, Any]]:
    rows = []
    for technique, report in reports.items():
        if report.get("status") == "failed":
            continue
        groups = {"id": report["its"]} if report["dataset"] == "its" else {
            "en": report["strict"]["en"], "it": report["strict"]["it"],
        }
        for language, group in groups.items():
            retrieval = group["retrieval"]
            rows.append({
                "technique": technique,
                "language": language,
                "BLEU_raw": group["BLEU_raw"]["score"],
                "BLEU_content": group["BLEU_content"]["score"],
                "exact_match": group["exact_match"],
                "token_f1": group["token_f1"],
                "recall_at_1": retrieval["recall_at_1"],
                "recall_at_5": retrieval["recall_at_5"],
                "recall_at_10": retrieval["recall_at_10"],
                "mrr_at_10": retrieval["mrr_at_10"],
                "ndcg_at_10": retrieval["ndcg_at_10"],
                "latency_mean_ms": group["latency_ms"]["end_to_end"]["mean"],
                "latency_p50_ms": group["latency_ms"]["end_to_end"]["p50"],
                "latency_p95_ms": group["latency_ms"]["end_to_end"]["p95"],
                "error_rate": group["error_rate"],
            })
    return rows


def _write_leaderboard(output: Path, reports: dict[str, dict]) -> None:
    rows = _leaderboard_rows(reports)
    write_json(output / "leaderboard.json", rows)
    with (output / "leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["technique"])
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(config_path: Path, stage: str, output: Path) -> dict[str, Any]:
    spec = ExperimentSpec.load(config_path)
    bundle = load_dataset(spec.dataset, spec.source)
    corpus_fingerprint = _corpus_fingerprint(bundle.documents)
    output.mkdir(parents=True, exist_ok=True)
    worker = RankifyWorkerClient(spec.rankify_python)
    capabilities = worker.invoke({"schema_version": 1, "operation": "capabilities"})
    result: dict[str, Any] = {"stage": stage, "index": None, "reports": None}

    if stage in {"index", "all"}:
        result["index"] = {
            "dillema": run_index("dillema", spec.dataset, spec.source, _dillema_index_root(spec)),
            "rankify": _index_rankify(spec, bundle, worker),
        }

    if stage in {"evaluate", "all"}:
        runtime = spec.generator.runtime()
        try:
            run_evaluate(
                "dillema", spec.dataset, _dillema_index_root(spec), output / "dillema",
                spec.source, runtime=runtime, model_fingerprint=spec.generator.fingerprint(),
                decoding_config=spec.generator.decoding_dict(),
            )
        finally:
            del runtime
            _release_gpu()
        dillema_report = summarize(read_predictions(output / "dillema" / "predictions.jsonl"))
        write_json(output / "dillema" / "metrics.json", dillema_report)
        reports = {"dillema": dillema_report}
        reports.update(_evaluate_rankify(spec, bundle, worker, output, corpus_fingerprint))
        _write_leaderboard(output, reports)
        result["reports"] = reports

    manifest = {
        "schema_version": 1,
        "created_at": _now(),
        "stage": stage,
        "experiment_fingerprint": spec.fingerprint(),
        "corpus_fingerprint": corpus_fingerprint,
        "config": spec.public_dict(),
        "rankify": capabilities,
        "result": result,
    }
    write_json(output / "manifest.json", manifest)
    return manifest
