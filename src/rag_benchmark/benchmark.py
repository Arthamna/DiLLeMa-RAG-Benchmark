from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .shared.artifacts import _corpus_fingerprint, _now
from .shared.config import ChunkingConfig, GenerationSpec
from .shared.datasets import load_dataset
from .shared.metrics import read_predictions, summarize, write_json
from .techniques.dillema.config import PipelineConfig
from .techniques.dillema.runner import run_evaluate, run_index
from .techniques.rankify.client import RankifyWorkerClient
from .techniques.rankify.config import RANKIFY_PRESETS
from .techniques.rankify.pipeline import _evaluate_rankify, _index_rankify
from .techniques.qwen3.config import Qwen3Config
from .techniques.qwen3.runner import run_index as index_qwen3, run_evaluate as evaluate_qwen3
from .techniques.bge_m3.config import BgeM3Config
from .techniques.bge_m3.runner import run_index as index_bge_m3, run_evaluate as evaluate_bge_m3

LOGGER = logging.getLogger(__name__)


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
    evaluation_split: str
    presets: tuple[str, ...]
    generator: GenerationSpec
    qwen3: Qwen3Config | None = None
    bge_m3: BgeM3Config | None = None

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
        evaluation = raw.get("evaluation") or {}
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
            evaluation_split=str(evaluation.get("split") or "test").strip().casefold(),
            presets=presets,
            generator=GenerationSpec.from_dict(raw.get("generator") or {}),
            qwen3=Qwen3Config.from_dict(raw.get("qwen3") or {}, base),
            bge_m3=BgeM3Config.from_dict(raw.get("bge_m3") or {}, base),
        ).validate()

    def validate(self) -> "ExperimentSpec":
        if not self.source.exists():
            raise FileNotFoundError(f"Dataset source not found: {self.source}")
        if self.generator.backend == "huggingface":
            model_path = Path(self.generator.model).expanduser()
            if not model_path.is_absolute():
                raise ValueError("Hugging Face local generator requires an absolute HF_MODEL_NAME snapshot path")
            if not (model_path / "config.json").is_file():
                raise FileNotFoundError(
                    f"Hugging Face snapshot config not found: {model_path / 'config.json'}"
                )
        if not self.dense_model or not self.reranker_model:
            raise ValueError("models.dense_multilingual and models.reranker_multilingual are required")
        if min(self.candidates, self.top_k, self.rrf_k) <= 0 or self.top_k > self.candidates:
            raise ValueError("retrieval values must be positive and top_k cannot exceed candidates")
        if self.evaluation_split not in {"test", "all"}:
            raise ValueError("evaluation.split must be test or all")
        return self

    def public_dict(self) -> dict[str, Any]:
        result = {
            "dataset": self.dataset,
            "source": str(self.source),
            "index_root": str(self.index_root),
            "rankify_python": str(self.rankify_python),
            "models": {"dense_multilingual": self.dense_model, "reranker_multilingual": self.reranker_model},
            "retrieval": {"candidates": self.candidates, "top_k": self.top_k, "rrf_k": self.rrf_k},
            "evaluation": {"split": self.evaluation_split},
            "presets": list(self.presets),
            "generator": self.generator.public_dict(),
        }
        if self.qwen3 is not None:
            result["qwen3"] = {"enabled": True, **self.qwen3.public_dict()}
        if self.bge_m3 is not None:
            result["bge_m3"] = {"enabled": True, **self.bge_m3.public_dict()}
        return result

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


def _dillema_index_root(spec: ExperimentSpec) -> Path:
    return spec.index_root / spec.index_fingerprint()[:16]


def _release_gpu(*, suppress_errors: bool = False) -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass
    except Exception as exc:
        if not suppress_errors:
            raise
        LOGGER.warning("CUDA cleanup dilewati setelah kegagalan sebelumnya: %s", exc)


def _log_cuda_runtime(spec: ExperimentSpec) -> None:
    try:
        revision = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unknown"

    try:
        import torch
    except ImportError:
        LOGGER.info("PyTorch tidak terpasang; CUDA tidak digunakan.")
        return

    if torch.cuda.is_available():
        LOGGER.info(
            "Runtime GPU: revision=%s source=%s torch=%s cuda=%s gpu=%s capability=%s archs=%s "
            "generator=%s quantization=%s",
            revision,
            Path(__file__).resolve(),
            torch.__version__,
            torch.version.cuda,
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_capability(0),
            torch.cuda.get_arch_list(),
            spec.generator.backend,
            spec.generator.quantization,
        )
    else:
        LOGGER.warning("PyTorch %s tidak mendeteksi CUDA.", torch.__version__)


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
    _log_cuda_runtime(spec)
    bundle = load_dataset(spec.dataset, spec.source)
    corpus_fingerprint = _corpus_fingerprint(bundle.documents)
    output.mkdir(parents=True, exist_ok=True)
    worker = RankifyWorkerClient(spec.rankify_python)
    capabilities = worker.invoke({"schema_version": 1, "operation": "capabilities"})
    result: dict[str, Any] = {"stage": stage, "index": None, "reports": None}

    if stage in {"index", "all"}:
        result["index"] = {
            "dillema": run_index(
                "dillema", spec.dataset, spec.source, _dillema_index_root(spec),
                pipeline_config=replace(PipelineConfig.from_env(), retrieval_candidates=spec.candidates),
            ),
        }
        _release_gpu()
        if spec.qwen3 is not None:
            result["index"]["qwen3"] = index_qwen3(spec, bundle)
        if spec.bge_m3 is not None:
            # All five BGE-M3 variants share this single physical index.
            result["index"]["bge_m3"] = index_bge_m3(spec, bundle)
        result["index"]["rankify"] = _index_rankify(spec, bundle, worker, ChunkingConfig())

    if stage in {"evaluate", "all"}:
        runtime = spec.generator.runtime()
        evaluation_failed = False
        try:
            run_evaluate(
                "dillema", spec.dataset, _dillema_index_root(spec), output / "dillema",
                spec.source, runtime=runtime, model_fingerprint=spec.generator.fingerprint(),
                decoding_config=spec.generator.decoding_dict(), evaluation_split=spec.evaluation_split,
                pipeline_config=replace(PipelineConfig.from_env(), retrieval_candidates=spec.candidates),
            )
        except BaseException:
            evaluation_failed = True
            raise
        finally:
            del runtime
            _release_gpu(suppress_errors=evaluation_failed)
        dillema_report = summarize(read_predictions(output / "dillema" / "predictions.jsonl"))
        write_json(output / "dillema" / "metrics.json", dillema_report)
        reports = {"dillema": dillema_report}
        if spec.qwen3 is not None:
            reports.update(evaluate_qwen3(spec, bundle, output))
        if spec.bge_m3 is not None:
            reports.update(evaluate_bge_m3(spec, bundle, output))
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
