from __future__ import annotations

import json
from pathlib import Path

from .artifacts import _now
from .metrics import read_predictions, summarize, write_json


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
        {row.get("corpus_fingerprint") for row in records},
        {json.dumps(row.get("decoding_config") or {}, sort_keys=True) for row in records},
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
        raise ValueError(
            "Technique runs must have identical dataset, split, model/corpus fingerprints, decoding, and QA IDs"
        )
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
