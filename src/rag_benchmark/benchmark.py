from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
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
from .techniques.self_rag.config import SelfRagConfig
from .techniques.self_rag.runner import run_index as index_self_rag, run_evaluate as evaluate_self_rag
from .techniques.rank_zephyr.config import RankZephyrConfig
from .techniques.rank_zephyr.runner import run_evaluate as evaluate_rank_zephyr, run_index as index_rank_zephyr


TECHNIQUE_GROUPS = ("dillema", "qwen3", "bge_m3", "self_rag", "rankify", "rank_zephyr")
SHARED_GENERATOR_GROUPS = frozenset({"dillema", "qwen3", "bge_m3", "rankify", "rank_zephyr"})


def parse_techniques(value) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = value.split(",") if isinstance(value, str) else list(value)
    requested = {str(item).strip().casefold() for item in items if str(item).strip()}
    unknown = requested - set(TECHNIQUE_GROUPS)
    if unknown or not requested:
        raise ValueError(f"techniques must be a comma-separated subset of: {', '.join(TECHNIQUE_GROUPS)}")
    return tuple(name for name in TECHNIQUE_GROUPS if name in requested)


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
    generator: GenerationSpec | None
    qwen3: Qwen3Config | None = None
    bge_m3: BgeM3Config | None = None
    self_rag: SelfRagConfig | None = None
    techniques: tuple[str, ...] = TECHNIQUE_GROUPS
    rank_zephyr: RankZephyrConfig | None = None

    @classmethod
    def load(cls, path: Path, techniques=None) -> "ExperimentSpec":
        requested = parse_techniques(techniques)
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
        qwen3 = Qwen3Config.from_dict(raw.get("qwen3") or {}, base)
        bge_m3 = BgeM3Config.from_dict(raw.get("bge_m3") or {}, base)
        self_rag = SelfRagConfig.from_dict(raw.get("self_rag") or {}, base)
        rank_zephyr = (
            RankZephyrConfig.from_dict(raw.get("rank_zephyr") or {}, base)
            if requested is None or "rank_zephyr" in requested else None
        )
        configured = {
            "dillema": True, "rankify": True, "qwen3": qwen3 is not None,
            "bge_m3": bge_m3 is not None, "self_rag": self_rag is not None,
            "rank_zephyr": rank_zephyr is not None,
        }
        if requested is None:
            selected = tuple(name for name in TECHNIQUE_GROUPS if configured[name])
        else:
            disabled = [name for name in requested if not configured[name]]
            if disabled:
                raise ValueError(f"Selected techniques are not enabled in the experiment YAML: {', '.join(disabled)}")
            selected = requested
        if "rankify" in selected and presets != RANKIFY_PRESETS:
            raise ValueError(f"presets must contain the controlled seven in order: {', '.join(RANKIFY_PRESETS)}")
        generator = (
            GenerationSpec.from_dict(raw.get("generator") or {})
            if SHARED_GENERATOR_GROUPS & set(selected) else None
        )
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
            generator=generator,
            qwen3=qwen3 if "qwen3" in selected else None,
            bge_m3=bge_m3 if "bge_m3" in selected else None,
            self_rag=self_rag if "self_rag" in selected else None,
            techniques=selected,
            rank_zephyr=rank_zephyr,
        ).validate()

    def validate(self) -> "ExperimentSpec":
        if not self.source.exists():
            raise FileNotFoundError(f"Dataset source not found: {self.source}")
        if self.generator is not None and self.generator.backend == "huggingface":
            model_path = Path(self.generator.model).expanduser()
            if not model_path.is_absolute():
                raise ValueError("Hugging Face local generator requires an absolute HF_MODEL_NAME snapshot path")
            if not (model_path / "config.json").is_file():
                raise FileNotFoundError(
                    f"Hugging Face snapshot config not found: {model_path / 'config.json'}"
                )
        if {"dillema", "rankify"} & set(self.techniques) and (not self.dense_model or not self.reranker_model):
            raise ValueError("models.dense_multilingual and models.reranker_multilingual are required")
        if min(self.candidates, self.top_k, self.rrf_k) <= 0 or self.top_k > self.candidates:
            raise ValueError("retrieval values must be positive and top_k cannot exceed candidates")
        if self.self_rag is not None and self.self_rag.ndocs > self.candidates:
            raise ValueError("self_rag.ndocs cannot exceed retrieval.candidates")
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
            "generator": self.generator.public_dict() if self.generator is not None else None,
            "techniques": list(self.techniques),
        }
        if self.qwen3 is not None:
            result["qwen3"] = {"enabled": True, **self.qwen3.public_dict()}
        if self.bge_m3 is not None:
            result["bge_m3"] = {"enabled": True, **self.bge_m3.public_dict()}
        if self.self_rag is not None:
            result["self_rag"] = {"enabled": True, **self.self_rag.public_dict()}
        if self.rank_zephyr is not None:
            result["rank_zephyr"] = {"enabled": True, **self.rank_zephyr.public_dict()}
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


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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


def run_benchmark(config_path: Path, stage: str, output: Path, techniques=None) -> dict[str, Any]:
    spec = ExperimentSpec.load(config_path, techniques)
    selected = set(spec.techniques)
    bundle = load_dataset(spec.dataset, spec.source)
    corpus_fingerprint = _corpus_fingerprint(bundle.documents)
    output.mkdir(parents=True, exist_ok=True)
    worker = capabilities = None
    if "rankify" in selected:
        worker = RankifyWorkerClient(spec.rankify_python)
        capabilities = worker.invoke({"schema_version": 1, "operation": "capabilities"})
    result: dict[str, Any] = {"stage": stage, "techniques": list(spec.techniques), "index": None, "reports": None}

    if stage in {"index", "all"}:
        result["index"] = {}
        if "dillema" in selected:
            result["index"]["dillema"] = run_index(
                "dillema", spec.dataset, spec.source, _dillema_index_root(spec),
                pipeline_config=replace(PipelineConfig.from_env(), retrieval_candidates=spec.candidates),
            )
            _release_gpu()
        if spec.qwen3 is not None:
            result["index"]["qwen3"] = index_qwen3(spec, bundle)
        if spec.bge_m3 is not None:
            # All five BGE-M3 variants share this single physical index.
            result["index"]["bge_m3"] = index_bge_m3(spec, bundle)
        if spec.self_rag is not None:
            result["index"]["self_rag"] = index_self_rag(spec, bundle)
        if worker is not None:
            result["index"]["rankify"] = _index_rankify(spec, bundle, worker, ChunkingConfig())
        if spec.rank_zephyr is not None:
            result["index"]["rank_zephyr"] = index_rank_zephyr(spec, bundle)

    if stage in {"evaluate", "all"}:
        reports = {}
        if "dillema" in selected:
            runtime = spec.generator.runtime()
            try:
                run_evaluate(
                    "dillema", spec.dataset, _dillema_index_root(spec), output / "dillema",
                    spec.source, runtime=runtime, model_fingerprint=spec.generator.fingerprint(),
                    decoding_config=spec.generator.decoding_dict(), evaluation_split=spec.evaluation_split,
                    pipeline_config=replace(PipelineConfig.from_env(), retrieval_candidates=spec.candidates),
                )
            finally:
                del runtime
                _release_gpu()
            reports["dillema"] = summarize(read_predictions(output / "dillema" / "predictions.jsonl"))
            write_json(output / "dillema" / "metrics.json", reports["dillema"])
        if spec.qwen3 is not None:
            reports.update(evaluate_qwen3(spec, bundle, output))
        if spec.bge_m3 is not None:
            reports.update(evaluate_bge_m3(spec, bundle, output))
        if spec.self_rag is not None:
            reports.update(evaluate_self_rag(spec, bundle, output))
        if worker is not None:
            reports.update(_evaluate_rankify(spec, bundle, worker, output, corpus_fingerprint))
        if spec.rank_zephyr is not None:
            reports.update(evaluate_rank_zephyr(spec, bundle, output))
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
