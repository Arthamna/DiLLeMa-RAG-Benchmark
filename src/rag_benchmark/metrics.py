from __future__ import annotations

import json
import re
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
    recalls = [
        float(row.get("diagnostics", {}).get("source_recall_at_10"))
        for row in records
        if row.get("diagnostics", {}).get("source_recall_at_10") is not None
    ]
    latencies = [float(row.get("latency_ms", 0.0)) for row in records]
    return {
        "count": len(records),
        "status_counts": dict(sorted(Counter(str(row.get("status", "")) for row in records).items())),
        "BLEU_raw": bleu_variant(records, content_only=False),
        "BLEU_content": bleu_variant(records, content_only=True),
        "retrieval": {
            "labeled_count": len(recalls),
            "recall_at_10": (sum(recalls) / len(recalls)) if recalls else None,
        },
        "average_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
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

