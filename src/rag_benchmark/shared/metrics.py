from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

from sacrebleu.metrics import BLEU

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_BOILERPLATE = re.compile(
    r"(?:"
    r"for\s+(?:more|further)\s+information"
    r"|per\s+(?:maggiori|ulteriori)\s+informazioni"
    r"|untuk\s+informasi\s+lebih\s+lanjut"
    r")[^.!?]*(?:[.!?]|$)",
    re.IGNORECASE,
)


def normalize_content(text: str) -> str:
    value = _URL.sub(" ", text or "")
    value = _BOILERPLATE.sub(" ", value)
    return " ".join(value.split()).strip()


def normalize_answer(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def _answer_scores(record: dict) -> tuple[float, float]:
    prediction = normalize_answer(str(record.get("prediction") or ""))
    references = [normalize_answer(str(item)) for item in record.get("references") or [""]]
    exact = 1.0 if prediction and prediction in references else 0.0
    prediction_tokens = prediction.split()
    best_f1 = 0.0
    for reference in references:
        reference_tokens = reference.split()
        if not prediction_tokens or not reference_tokens:
            continue
        overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
        if not overlap:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return exact, best_f1


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _latency(values: list[float]) -> dict:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _retrieval_scores(records: list[dict]) -> dict:
    labeled = [row for row in records if row.get("gold_source_ids")]
    labels = sorted({
        str(row.get("diagnostics", {}).get("relevance_label")) for row in labeled
        if row.get("diagnostics", {}).get("relevance_label")
    })
    recalls = {1: [], 5: [], 10: []}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for row in labeled:
        ranks = [int(value) for value in row.get("diagnostics", {}).get("relevant_ranks", [])]
        for cutoff in recalls:
            recalls[cutoff].append(1.0 if any(rank <= cutoff for rank in ranks) else 0.0)
        first = min(ranks) if ranks else None
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= 10)
        ideal_count = min(len(set(row.get("gold_source_ids") or [])), 10)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    result = {
        "labeled_count": len(labeled),
        "relevance_label": labels[0] if len(labels) == 1 else labels,
        "recall_at_1": sum(recalls[1]) / len(labeled) if labeled else None,
        "recall_at_5": sum(recalls[5]) / len(labeled) if labeled else None,
        "recall_at_10": sum(recalls[10]) / len(labeled) if labeled else None,
        "mrr_at_10": sum(reciprocal_ranks) / len(labeled) if labeled else None,
        "ndcg_at_10": sum(ndcgs) / len(labeled) if labeled else None,
    }
    if any(row.get("diagnostics", {}).get("candidate_cutoff") == 100 for row in records):
        result["candidate_recall_at_100"] = (
            sum(bool(row.get("diagnostics", {}).get("candidate_relevant_ranks")) for row in labeled)
            / len(labeled) if labeled else None
        )
    return result


def _reference_streams(records: list[dict], transform) -> list[list[str]]:
    width = max((len(record.get("references") or []) for record in records), default=1)
    streams: list[list[str]] = [[] for _ in range(max(1, width))]
    for record in records:
        references = [str(item) for item in record.get("references") or [""]]
        primary = references[0] if references else ""
        for index, stream in enumerate(streams):
            reference = references[index] if index < len(references) and references[index] else primary
            stream.append(transform(reference))
    return streams


def bleu_variant(records: list[dict], content_only: bool = False) -> dict:
    transform = normalize_content if content_only else (lambda text: text or "")
    predictions = [transform(str(record.get("prediction") or "")) for record in records]
    references = _reference_streams(records, transform)
    metric = BLEU(lowercase=False, tokenize="13a", effective_order=False)
    result = metric.corpus_score(predictions, references)
    return {
        "score": round(float(result.score), 10),
        "signature": str(metric.get_signature()),
        "counts": list(result.counts),
        "totals": list(result.totals),
        "precisions": [float(value) for value in result.precisions],
        "bp": float(result.bp),
        "sys_len": int(result.sys_len),
        "ref_len": int(result.ref_len),
    }


def score_group(records: list[dict]) -> dict:
    answer_scores = [_answer_scores(row) for row in records]
    latencies = [float(row.get("latency_ms", 0.0)) for row in records]
    stage_latency = {}
    for key in ("retrieval_latency_ms", "rerank_latency_ms", "generation_latency_ms"):
        values = [
            float(row.get("diagnostics", {}).get(key)) for row in records
            if row.get("diagnostics", {}).get(key) is not None
        ]
        stage_latency[key.removesuffix("_latency_ms")] = _latency(values)
    statuses = Counter(str(row.get("status", "")) for row in records)
    return {
        "count": len(records),
        "status_counts": dict(sorted(statuses.items())),
        "BLEU_raw": bleu_variant(records, content_only=False),
        "BLEU_content": bleu_variant(records, content_only=True),
        "exact_match": sum(score[0] for score in answer_scores) / len(records) if records else 0.0,
        "token_f1": sum(score[1] for score in answer_scores) / len(records) if records else 0.0,
        "retrieval": _retrieval_scores(records),
        "average_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "latency_ms": {"end_to_end": _latency(latencies), **stage_latency},
        "error_rate": statuses.get("error", 0) / len(records) if records else 0.0,
        "cpu_fallback_count": sum(
            1 for row in records if row.get("diagnostics", {}).get("gpu_fallback")
        ),
    }


def _macro(en: dict, it: dict) -> dict:
    return {
        "BLEU_raw": (en["BLEU_raw"]["score"] + it["BLEU_raw"]["score"]) / 2,
        "BLEU_content": (en["BLEU_content"]["score"] + it["BLEU_content"]["score"]) / 2,
        "languages": ["en", "it"],
    }


def summarize(records: list[dict]) -> dict:
    datasets = {str(record.get("dataset")) for record in records}
    if datasets == {"its"} or (not records and not datasets):
        return {"dataset": "its", "its": score_group(records)}
    if datasets != {"uniqa"}:
        raise ValueError(f"Expected records from one dataset, got: {sorted(datasets)}")

    strict_en = [row for row in records if row.get("language") == "en"]
    strict_it = [row for row in records if row.get("language") == "it"]
    valid_en = [row for row in strict_en if bool(row.get("available", True))]
    valid_it = [row for row in strict_it if bool(row.get("available", True))]
    strict = {"en": score_group(strict_en), "it": score_group(strict_it)}
    valid = {"en": score_group(valid_en), "it": score_group(valid_it)}
    strict["macro_average"] = _macro(strict["en"], strict["it"])
    valid["macro_average"] = _macro(valid["en"], valid["it"])
    return {"dataset": "uniqa", "strict": strict, "corpus_valid": valid}


def read_predictions(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

