from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .shared.comparison import run_compare, run_compare_techniques
from .techniques import TECHNIQUES

LOGGER = logging.getLogger(__name__)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _launch_detached_benchmark(config: Path, stage: str, output: Path, techniques: str | None = None) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("Detached benchmark execution is supported only on Linux")
    from .benchmark import ExperimentSpec

    config = config.resolve()
    ExperimentSpec.load(config, techniques)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path, pid_path = output / "benchmark.log", output / "benchmark.pid"
    if pid_path.is_file():
        try:
            existing_pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            existing_pid = 0
        if _pid_is_running(existing_pid):
            raise RuntimeError(f"A detached benchmark is already running with PID {existing_pid}: {pid_path}")
    command = [
        sys.executable, "-u", "-m", "rag_benchmark", "benchmark", "--config", str(config),
        "--stage", stage, "--output", str(output),
    ]
    if techniques:
        command += ["--techniques", techniques]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()), env=environment, start_new_session=True, close_fds=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return {"status": "started", "pid": process.pid, "log": str(log_path),
            "pid_file": str(pid_path), "output": str(output)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag_benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    techniques = tuple(TECHNIQUES)
    index = subparsers.add_parser("index", help="Build a technique index")
    index.add_argument("--technique", choices=techniques, required=True)
    index.add_argument("--dataset", choices=("its", "uniqa"), required=True)
    index.add_argument("--source", type=Path)
    index.add_argument("--index-root", type=Path)
    index.add_argument("--config", type=Path)
    evaluate = subparsers.add_parser("evaluate", help="Evaluate one technique")
    evaluate.add_argument("--technique", choices=techniques, required=True)
    evaluate.add_argument("--dataset", choices=("its", "uniqa"), required=True)
    evaluate.add_argument("--index-root", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--source", type=Path)
    evaluate.add_argument("--config", type=Path)
    compare = subparsers.add_parser("compare", help="Compare completed ITS and UniQA runs")
    compare.add_argument("--its-run", type=Path, required=True)
    compare.add_argument("--uniqa-run", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    technique_compare = subparsers.add_parser("compare-techniques", help="Compare two technique runs")
    technique_compare.add_argument("--baseline-run", type=Path, required=True)
    technique_compare.add_argument("--candidate-run", type=Path, required=True)
    technique_compare.add_argument("--output", type=Path, required=True)
    benchmark = subparsers.add_parser("benchmark", help="Run the configured technique matrix")
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--stage", choices=("index", "evaluate", "all"), required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--techniques", help="Comma-separated subset of dillema,bge_m3,self_rag,rank_zephyr,retro,piperag")
    benchmark.add_argument("-d", "--detach", action="store_true")
    doctor = subparsers.add_parser("doctor", help="Validate services, assets, protocol, index and GPU")
    doctor.add_argument("--config", type=Path, required=True)
    return parser


def _single_run(args, parser):
    from .benchmark import _gpu_lease, _gpu_lock_root

    with _gpu_lease(_gpu_lock_root(), args.output if args.command == "evaluate" else Path.cwd()):
        return _single_run_locked(args, parser)


def _single_run_locked(args, parser):
    from .benchmark import ExperimentSpec, _dillema_index_root, _local_runner
    from .shared.datasets import load_dataset
    from .techniques.dillema.config import PipelineConfig
    from .techniques.dillema.runner import run_evaluate as evaluate_dillema
    from .techniques.dillema.runner import run_index as index_dillema
    from .techniques.retro.runner import run_evaluate as evaluate_retro
    from .techniques.retro.runner import run_index as index_retro

    if args.technique == "dillema" and args.config is None:
        if args.source is None:
            parser.error("DiLLeMa index/evaluate without --config requires --source")
        root = args.index_root or Path("artifacts/indexes")
        if args.command == "index":
            return index_dillema("dillema", args.dataset, args.source, root)
        return evaluate_dillema("dillema", args.dataset, root, args.output, args.source)
    if args.config is None:
        parser.error("This technique requires --config")
    group = ("bge_m3" if args.technique.startswith("bge_m3_") else
             "self_rag" if args.technique.startswith("self_rag_") else args.technique)
    spec = ExperimentSpec.load(args.config, group)
    if args.dataset != spec.dataset:
        parser.error("--dataset must match the experiment YAML")
    spec = replace(spec, source=args.source or spec.source, index_root=args.index_root or spec.index_root)
    bundle = load_dataset(spec.dataset, spec.source)
    if group in {"bge_m3", "self_rag"}:
        runner = _local_runner(group)
        if args.command == "index":
            return runner.run_index(spec, bundle)
        return runner.run_evaluate(spec, bundle, args.output, (args.technique,))
    if args.command == "index":
        if args.technique == "dillema":
            return index_dillema(
                "dillema", spec.dataset, spec.source, _dillema_index_root(spec),
                pipeline_config=replace(PipelineConfig.from_env(), retrieval_candidates=spec.candidates),
            )
        return index_retro(spec, bundle)
    if args.technique == "dillema":
        return evaluate_dillema(
            "dillema", spec.dataset, _dillema_index_root(spec), args.output, spec.source,
            runtime_factory=spec.generator.runtime, model_fingerprint=spec.generator.fingerprint(),
            decoding_config=spec.generator.decoding_dict(), evaluation_split=spec.evaluation_split,
            pipeline_config=replace(PipelineConfig.from_env(), retrieval_candidates=spec.candidates),
        )
    return evaluate_retro(spec, bundle, args.output, (args.technique,))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"index", "evaluate"}:
        result = _single_run(args, parser)
    elif args.command == "compare":
        result = run_compare(args.its_run, args.uniqa_run, args.output)
    elif args.command == "compare-techniques":
        result = run_compare_techniques(args.baseline_run, args.candidate_run, args.output)
    elif args.command == "doctor":
        from .benchmark import ExperimentSpec
        from .techniques.retro.runner import doctor
        result = doctor(ExperimentSpec.load(args.config, "retro,piperag"))
    else:
        from .benchmark import run_benchmark
        result = (_launch_detached_benchmark(args.config, args.stage, args.output, args.techniques)
                  if args.detach else run_benchmark(args.config, args.stage, args.output, args.techniques))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
