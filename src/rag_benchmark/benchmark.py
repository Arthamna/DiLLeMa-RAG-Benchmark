from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from .shared.artifacts import _corpus_fingerprint, _now
from .shared.config import GenerationSpec
from .shared.datasets import load_dataset
from .shared.device import release_cuda_memory
from .shared.metrics import read_predictions, summarize, write_json
from .techniques.dillema.config import PipelineConfig
from .techniques.dillema.runner import run_evaluate as evaluate_dillema
from .techniques.dillema.runner import run_index as index_dillema
from .techniques.piperag.config import PipeRagConfig
from .techniques.retro.config import PaperEvaluationConfig, RetroConfig, ServiceConfig
from .techniques.retro.runner import run_evaluate as evaluate_retro
from .techniques.retro.runner import run_index as index_retro
from .techniques.retro.runner import run_paper_evaluation
from .techniques.retro.client import InferenceServiceClient
from .techniques.bge_m3.config import BgeM3Config
from .techniques.self_rag.config import SelfRagConfig
from .techniques.rank_zephyr.config import RankZephyrConfig

DEFAULT_TECHNIQUES = ("dillema", "retro", "piperag")
TECHNIQUE_GROUPS = ("dillema", "bge_m3", "self_rag", "rank_zephyr", "retro", "piperag")
SHARED_GENERATOR_GROUPS = frozenset({"dillema", "bge_m3", "rank_zephyr"})
LOCAL_GROUPS = ("bge_m3", "self_rag", "rank_zephyr")
LOGGER = logging.getLogger(__name__)


def parse_techniques(value) -> tuple[str, ...] | None:
    if value is None:
        return None
    items = value.split(",") if isinstance(value, str) else list(value)
    requested = {str(item).strip().casefold() for item in items if str(item).strip()}
    unknown = requested - set(TECHNIQUE_GROUPS)
    if unknown or not requested:
        raise ValueError(f"techniques must be a comma-separated subset of: {', '.join(TECHNIQUE_GROUPS)}")
    return tuple(name for name in TECHNIQUE_GROUPS if name in requested)


def _resolved(base: Path, value: Any, field: str, default: str | None = None) -> Path:
    raw = str(value or default or "").strip()
    if not raw:
        raise ValueError(f"Missing required configuration: {field}")
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    dataset: str
    source: Path
    index_root: Path
    evaluation_split: str
    candidates: int
    generator: GenerationSpec | None
    retro: RetroConfig | None
    piperag: PipeRagConfig | None
    services: ServiceConfig | None
    paper: PaperEvaluationConfig
    techniques: tuple[str, ...] = DEFAULT_TECHNIQUES
    top_k: int = 10
    bge_m3: BgeM3Config | None = None
    self_rag: SelfRagConfig | None = None
    rank_zephyr: RankZephyrConfig | None = None

    @classmethod
    def load(cls, path: Path, techniques=None) -> "ExperimentSpec":
        requested = parse_techniques(techniques)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        dataset = str(raw.get("dataset") or "").casefold()
        if dataset not in {"its", "uniqa"}:
            raise ValueError("dataset must be its or uniqa")
        base = path.resolve().parent
        configured = raw.get("techniques")
        selected = requested or parse_techniques(DEFAULT_TECHNIQUES if configured is None else configured)
        remote_selected = bool({"retro", "piperag"} & set(selected))
        local_configs = {}
        for name, config_type in (("bge_m3", BgeM3Config), ("self_rag", SelfRagConfig),
                                  ("rank_zephyr", RankZephyrConfig)):
            if name in selected:
                config = config_type.from_dict(raw.get(name) or {}, base)
                if config is None:
                    raise ValueError(f"Selected technique is not enabled in the experiment YAML: {name}")
                local_configs[name] = config
        generator = (GenerationSpec.from_dict(raw.get("generator") or {})
                     if SHARED_GENERATOR_GROUPS & set(selected) else None)
        retro = RetroConfig.from_dict(raw.get("retro") or {}, base) if remote_selected else None
        piperag = PipeRagConfig.from_dict(raw.get("piperag") or {}) if "piperag" in selected else None
        services = ServiceConfig.from_dict(raw.get("services") or {}) if remote_selected else None
        evaluation = raw.get("evaluation") or {}
        retrieval = raw.get("retrieval") or {}
        return cls(
            dataset=dataset,
            source=_resolved(base, raw.get("source"), "source"),
            index_root=_resolved(base, raw.get("index_root"), "index_root", "../../artifacts/indexes"),
            evaluation_split=str(evaluation.get("split") or "test").strip().casefold(),
            candidates=int(retrieval.get("candidates", 40)),
            generator=generator,
            retro=retro,
            piperag=piperag,
            services=services,
            paper=PaperEvaluationConfig.from_dict((raw.get("paper_evaluation") or {}) if remote_selected else {}),
            techniques=selected,
            top_k=int(retrieval.get("top_k", 10)),
            **local_configs,
        ).validate()

    def validate(self) -> "ExperimentSpec":
        if not self.source.exists():
            raise FileNotFoundError(f"Dataset source not found: {self.source}")
        if self.evaluation_split not in {"test", "all"}:
            raise ValueError("evaluation.split must be test or all")
        if self.candidates <= 0:
            raise ValueError("retrieval.candidates must be positive")
        if {"bge_m3", "rank_zephyr"} & set(self.techniques) and not 0 < self.top_k <= self.candidates:
            raise ValueError("retrieval.top_k must be positive and cannot exceed candidates")
        if self.self_rag is not None and self.self_rag.ndocs > self.candidates:
            raise ValueError("self_rag.ndocs cannot exceed retrieval.candidates")
        if self.generator is not None:
            if self.generator.max_tokens != 128:
                raise ValueError("Official benchmark requires generator.max_tokens=128")
            if self.generator.backend == "huggingface":
                model = Path(self.generator.model).expanduser()
                if not model.is_absolute() or not (model / "config.json").is_file():
                    raise FileNotFoundError(f"Hugging Face snapshot config not found: {model / 'config.json'}")
        if {"retro", "piperag"} & set(self.techniques) and (self.retro is None or self.services is None):
            raise ValueError("RETRO/PipeRAG require retro and services configuration")
        if "piperag" in self.techniques and self.piperag is None:
            raise ValueError("PipeRAG requires piperag configuration")
        return self

    def public_dict(self) -> dict[str, Any]:
        result = {
            "dataset": self.dataset, "source": str(self.source), "index_root": str(self.index_root),
            "evaluation": {"split": self.evaluation_split}, "retrieval": {"candidates": self.candidates},
            "generator": self.generator.public_dict() if self.generator else None,
            "retro": self.retro.public_dict() if self.retro else None,
            "piperag": self.piperag.public_dict() if self.piperag else None,
            "services": asdict(self.services) if self.services else None,
            "paper_evaluation": asdict(self.paper), "techniques": list(self.techniques),
        }
        if {"bge_m3", "rank_zephyr"} & set(self.techniques):
            result["retrieval"]["top_k"] = self.top_k
        for name in LOCAL_GROUPS:
            config = getattr(self, name)
            if config is not None:
                result[name] = {"enabled": True, **config.public_dict()}
        return result

    def fingerprint(self) -> str:
        payload = json.dumps(self.public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def index_fingerprint(self) -> str:
        pipeline = asdict(PipelineConfig.from_env())
        payload = {
            "dataset": self.dataset, "source": str(self.source), "retrieval_candidates": self.candidates,
            "chunking": {key: pipeline[key] for key in (
                "chunk_size", "chunk_overlap", "min_chunk_chars", "parent_window_chars",
                "embedding_model", "reranker_model",
            )},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dillema_index_root(spec: ExperimentSpec) -> Path:
    return spec.index_root / "dillema" / spec.index_fingerprint()[:16]


def _local_runner(name: str):
    """Import only the selected technique's runtime dependencies."""
    from importlib import import_module

    if name not in LOCAL_GROUPS:
        raise ValueError(f"Unsupported local technique group: {name}")
    return import_module(f".techniques.{name}.runner", __package__)


def _gpu_lock_root() -> Path:
    # A namespace-specific index root must not create an independent GPU lease.
    return Path(tempfile.gettempdir()) / f"rag-benchmark-{os.getuid()}"


@contextmanager
def _gpu_lease(root: Path, output: Path):
    """Prevent concurrent benchmark processes from oversubscribing one GPU."""

    root.mkdir(parents=True, exist_ok=True)
    path = root / ".benchmark-gpu.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown process"
            raise RuntimeError(f"GPU benchmark is already active ({owner}); wait for it or stop that process") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} output={output.resolve()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _unload_remote_inference(spec: ExperimentSpec) -> None:
    if spec.services is None:
        return
    client = InferenceServiceClient(spec.services)
    try:
        client.unload()
        LOGGER.info("RETRO/PipeRAG inference GPU state unloaded.")
    except Exception as exc:
        LOGGER.warning("Could not unload inference service before the next GPU phase: %s", exc)
    finally:
        client.close()


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
                "technique": technique, "language": language,
                "BLEU_raw": group["BLEU_raw"]["score"], "BLEU_content": group["BLEU_content"]["score"],
                "exact_match": group["exact_match"], "token_f1": group["token_f1"],
                "recall_at_1": retrieval["recall_at_1"], "recall_at_5": retrieval["recall_at_5"],
                "recall_at_10": retrieval["recall_at_10"], "mrr_at_10": retrieval["mrr_at_10"],
                "ndcg_at_10": retrieval["ndcg_at_10"],
                "latency_mean_ms": group["latency_ms"]["end_to_end"]["mean"],
                "latency_p50_ms": group["latency_ms"]["end_to_end"]["p50"],
                "latency_p95_ms": group["latency_ms"]["end_to_end"]["p95"], "error_rate": group["error_rate"],
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
    with _gpu_lease(_gpu_lock_root(), output):
        return _run_benchmark_locked(spec, stage, output)


def _run_benchmark_locked(spec: ExperimentSpec, stage: str, output: Path) -> dict[str, Any]:
    selected = set(spec.techniques)
    remote = tuple(name for name in ("retro", "piperag") if name in selected)
    bundle = load_dataset(spec.dataset, spec.source)
    corpus_fingerprint = _corpus_fingerprint(bundle.documents)
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"stage": stage, "techniques": list(spec.techniques), "index": None, "reports": None}
    if (SHARED_GENERATOR_GROUPS | {"self_rag"}) & selected and remote:
        _unload_remote_inference(spec)
    if stage in {"index", "all"}:
        result["index"] = {}
        if "dillema" in selected:
            result["index"]["dillema"] = index_dillema(
                "dillema", spec.dataset, spec.source, _dillema_index_root(spec),
                pipeline_config=replace(PipelineConfig.from_env(), retrieval_candidates=spec.candidates),
            )
            release_cuda_memory("DiLLeMa indexing phase")
        for name in LOCAL_GROUPS:
            if name in selected:
                try:
                    result["index"][name] = _local_runner(name).run_index(spec, bundle)
                finally:
                    release_cuda_memory(f"{name} indexing phase")
        if remote:
            result["index"]["retro_piperag"] = index_retro(spec, bundle)
    if stage in {"evaluate", "all"}:
        reports: dict[str, dict] = {}
        if "dillema" in selected:
            evaluate_dillema(
                "dillema", spec.dataset, _dillema_index_root(spec), output / "dillema",
                spec.source, runtime_factory=spec.generator.runtime,
                model_fingerprint=spec.generator.fingerprint(),
                decoding_config=spec.generator.decoding_dict(), evaluation_split=spec.evaluation_split,
                pipeline_config=replace(PipelineConfig.from_env(), retrieval_candidates=spec.candidates),
            )
            reports["dillema"] = summarize(read_predictions(output / "dillema" / "predictions.jsonl"))
            write_json(output / "dillema" / "metrics.json", reports["dillema"])
        for name in LOCAL_GROUPS:
            if name in selected:
                try:
                    reports.update(_local_runner(name).run_evaluate(spec, bundle, output))
                finally:
                    release_cuda_memory(f"{name} evaluation phase")
        if remote:
            try:
                reports.update(evaluate_retro(spec, bundle, output, remote))
                result["paper"] = run_paper_evaluation(spec, bundle, output, remote)
            finally:
                _unload_remote_inference(spec)
        _write_leaderboard(output, reports)
        result["reports"] = reports
    manifest = {
        "schema_version": 3, "created_at": _now(), "stage": stage,
        "experiment_fingerprint": spec.fingerprint(), "corpus_fingerprint": corpus_fingerprint,
        "config": spec.public_dict(),
        "comparability": {
            **({"dillema_vs_retro": "cross-model", "retro_vs_piperag": "controlled-same-checkpoint"}
               if remote else {}),
            **({"self_rag_vs_shared_generator": "cross-model"}
               if "self_rag" in selected and SHARED_GENERATOR_GROUPS & selected else {}),
        },
        "result": result,
    }
    write_json(output / "manifest.json", manifest)
    return manifest
