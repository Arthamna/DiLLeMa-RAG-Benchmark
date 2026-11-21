"""GPU smoke harness: evaluate two available test questions per language.

Build the complete indexes first with benchmark --stage index. This harness
patches only dataset loading during evaluation; corpus/index identity stays
unchanged. Smoke outputs are explicitly marked and must not be used as full runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rag_benchmark import benchmark
from rag_benchmark.shared.datasets import load_dataset
from rag_benchmark.shared.metrics import read_predictions, write_json
from rag_benchmark.techniques.bge_m3.config import VARIANTS as BGE_VARIANTS
from rag_benchmark.techniques.self_rag.config import VARIANTS as SELF_VARIANTS
from rag_benchmark.techniques.rank_zephyr.config import VARIANTS as RZ_VARIANTS

GROUPS = "dillema,bge_m3,self_rag,rank_zephyr"
VARIANTS = ("dillema", *BGE_VARIANTS, *SELF_VARIANTS, *RZ_VARIANTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        import torch
    except ImportError:
        parser.error("Torch is not installed; run scripts/setup_ubuntu.sh on the GPU host first")
    if not torch.cuda.is_available():
        parser.error("This smoke test requires an NVIDIA CUDA GPU")
    os.environ["RAG_DEVICE"] = "cuda"
    spec = benchmark.ExperimentSpec.load(args.config, GROUPS)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory dedicated to smoke results")
    worker = Path(__file__).resolve().parents[1] / "src/rag_benchmark/techniques/rank_zephyr/worker.py"
    bundle = load_dataset(spec.dataset, spec.source)
    available = [qa for qa in bundle.questions_for_split(spec.evaluation_split) if qa.available]
    languages = ("id",) if spec.dataset == "its" else ("en", "it")
    sample = []
    for language in languages:
        selected = [qa for qa in available if qa.language == language][:2]
        if len(selected) != 2:
            parser.error(f"Expected at least two available QA for {language}")
        sample.extend(selected)
    sampled_bundle = replace(bundle, questions=sample)
    expected_ids = {qa.id for qa in sample}
    args.output.mkdir(parents=True, exist_ok=True)
    marker = {"smoke_test": True, "limit_per_language": 2, "qa_ids": sorted(expected_ids)}
    write_json(args.output / "smoke.json", {**marker, "status": "running"})
    try:
        with benchmark._gpu_lease(benchmark._gpu_lock_root(), args.output):
            subprocess.run([spec.rank_zephyr.worker_python, "-I", str(worker), "--probe"], check=True)
            with patch.object(benchmark, "load_dataset", return_value=sampled_bundle), patch(
                "rag_benchmark.techniques.dillema.runner.load_dataset", return_value=sampled_bundle,
            ):
                manifest = benchmark._run_benchmark_locked(spec, "evaluate", args.output)
        manifest.update(marker)
        write_json(args.output / "manifest.json", manifest)
        if set(manifest["result"]["reports"]) != set(VARIANTS):
            raise RuntimeError("Smoke run did not produce all ten variants")
        predictions = {}
        for variant in VARIANTS:
            report = manifest["result"]["reports"][variant]
            if report.get("status") == "failed":
                raise RuntimeError(f"{variant} failed: {report.get('error')}")
            rows = read_predictions(args.output / variant / "predictions.jsonl")
            if len(rows) != len(sample) or {row["id"] for row in rows} != expected_ids:
                raise RuntimeError(f"{variant}: sample QA IDs/count differ")
            if any(row["status"] not in {"ok", "abstention", "no_evidence"} for row in rows):
                raise RuntimeError(f"{variant}: prediction errors; inspect predictions.jsonl")
            predictions[variant] = {row["id"]: row for row in rows}
            child_path = args.output / variant / "manifest.json"
            child = json.loads(child_path.read_text())
            child.update(marker)
            write_json(child_path, child)
        for names in (BGE_VARIANTS, RZ_VARIANTS):
            for qid in expected_ids:
                hashes = {predictions[name][qid]["diagnostics"]["candidate_pool_hash"] for name in names}
                if len(hashes) != 1:
                    raise RuntimeError(f"Candidate pools differ for {names}, QA {qid}")
        for row in predictions["rank_zephyr_rerank"].values():
            diagnostics = row["diagnostics"]
            # Small corpora may have fewer candidates than the requested cutoff.
            count = diagnostics["retrieved_candidates"]
            expected = 0 if count == 0 else max(1, (max(0, count - 20) + 9) // 10 + 1)
            if diagnostics["rerank_windows"] != expected:
                raise RuntimeError(f"Unexpected reranking window count for {row['id']}")
        write_json(args.output / "smoke.json", {**marker, "status": "passed"})
    except Exception as exc:
        write_json(args.output / "smoke.json", {**marker, "status": "failed", "error": str(exc)})
        raise
    print(f"GPU smoke passed: {len(sample)} QA, {len(VARIANTS)} variants. Results: {args.output}")


if __name__ == "__main__":
    main()
