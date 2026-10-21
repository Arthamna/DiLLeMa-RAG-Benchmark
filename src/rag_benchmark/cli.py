from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

from .shared.comparison import run_compare, run_compare_techniques
from .techniques import TECHNIQUES
from .techniques.dillema.runner import run_evaluate, run_index

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


def _launch_detached_benchmark(config: Path, stage: str, output: Path) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("Detached benchmark execution is supported only on Linux")

    from .benchmark import ExperimentSpec

    config = config.resolve()
    ExperimentSpec.load(config)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "benchmark.log"
    pid_path = output / "benchmark.pid"

    if pid_path.is_file():
        try:
            existing_pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            existing_pid = 0
        if _pid_is_running(existing_pid):
            raise RuntimeError(
                f"A detached benchmark is already running with PID {existing_pid}: {pid_path}"
            )

    command = [
        sys.executable,
        "-u",
        "-m",
        "rag_benchmark",
        "benchmark",
        "--config",
        str(config),
        "--stage",
        stage,
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            env=environment,
            start_new_session=True,
            close_fds=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return {
        "status": "started",
        "pid": process.pid,
        "log": str(log_path),
        "pid_file": str(pid_path),
        "output": str(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag_benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    techniques = tuple(TECHNIQUES)

    index = subparsers.add_parser("index", help="Build a technique-isolated RAG index")
    index.add_argument("--technique", choices=techniques, required=True)
    index.add_argument("--dataset", choices=("its", "uniqa"), required=True)
    index.add_argument("--source", type=Path, required=True)
    index.add_argument("--index-root", type=Path)
    index.add_argument("--config", type=Path, help="Experiment YAML; required for Qwen3 techniques")

    evaluate = subparsers.add_parser("evaluate", help="Run technique generation and BLEU")
    evaluate.add_argument("--technique", choices=techniques, required=True)
    evaluate.add_argument("--dataset", choices=("its", "uniqa"), required=True)
    evaluate.add_argument("--index-root", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--source", type=Path)
    evaluate.add_argument("--config", type=Path, help="Experiment YAML; required for Qwen3 techniques")

    compare = subparsers.add_parser("compare", help="Compare completed ITS and UniQA runs")
    compare.add_argument("--its-run", type=Path, required=True)
    compare.add_argument("--uniqa-run", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    techniques_compare = subparsers.add_parser("compare-techniques", help="Compare controlled technique runs")
    techniques_compare.add_argument("--baseline-run", type=Path, required=True)
    techniques_compare.add_argument("--candidate-run", type=Path, required=True)
    techniques_compare.add_argument("--output", type=Path, required=True)

    benchmark = subparsers.add_parser("benchmark", help="Run the controlled DiLLeMa, Rankify and Qwen3 matrix")
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--stage", choices=("index", "evaluate", "all"), required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("-d", "--detach", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"index", "evaluate"} and args.technique.startswith("qwen3_"):
        from dataclasses import replace
        from .benchmark import ExperimentSpec
        from .shared.datasets import load_dataset
        from .techniques.qwen3.runner import run_evaluate as evaluate_qwen3, run_index as index_qwen3

        if args.config is None:
            parser.error("Qwen3 techniques require --config with an enabled qwen3 block")
        spec = ExperimentSpec.load(args.config)
        if spec.qwen3 is None or args.dataset != spec.dataset:
            parser.error("Qwen3 must be enabled and --dataset must match the experiment YAML")
        spec = replace(spec, source=args.source or spec.source, index_root=args.index_root or spec.index_root)
        bundle = load_dataset(spec.dataset, spec.source)
        result = (index_qwen3(spec, bundle) if args.command == "index" else
                  evaluate_qwen3(spec, bundle, args.output, variants=(args.technique,)))
    elif args.command == "index":
        result = run_index(args.technique, args.dataset, args.source, args.index_root or Path("artifacts/indexes"))
    elif args.command == "evaluate":
        result = run_evaluate(args.technique, args.dataset, args.index_root or Path("artifacts/indexes"), args.output, args.source)
    elif args.command == "compare":
        result = run_compare(args.its_run, args.uniqa_run, args.output)
    elif args.command == "compare-techniques":
        result = run_compare_techniques(args.baseline_run, args.candidate_run, args.output)
    else:
        from .benchmark import run_benchmark

        if args.detach:
            result = _launch_detached_benchmark(args.config, args.stage, args.output)
        else:
            result = run_benchmark(args.config, args.stage, args.output)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
