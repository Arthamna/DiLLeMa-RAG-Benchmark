from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from ...shared.records import DocumentRecord


def choose_nlist(num_chunks: int) -> int:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    target = min(1024, max(1, int(math.sqrt(num_chunks))))
    return 1 << (target.bit_length() - 1)


def clamp_nprobe(value: int, nlist: int) -> int:
    if nlist <= 0:
        raise ValueError("nlist must be positive")
    return max(1, min(int(value), int(nlist)))


def deterministic_holdout(
    documents: list[DocumentRecord], fraction: float = 0.10, seed: int = 42,
) -> tuple[list[DocumentRecord], list[DocumentRecord]]:
    by_language: dict[str, list[DocumentRecord]] = defaultdict(list)
    for document in documents:
        by_language[document.language].append(document)
    train: list[DocumentRecord] = []
    holdout: list[DocumentRecord] = []
    for language in sorted(by_language):
        values = sorted(
            by_language[language],
            key=lambda item: hashlib.sha256(f"{seed}:{item.id}".encode()).hexdigest(),
        )
        if len(values) == 1:
            holdout_count = 1
        else:
            holdout_count = min(len(values) - 1, max(1, math.ceil(len(values) * fraction)))
        holdout.extend(values[:holdout_count])
        train.extend(values[holdout_count:])
    return train, holdout


def document_payload(document: DocumentRecord) -> dict:
    return {
        "id": document.id,
        "dataset": document.dataset,
        "language": document.language,
        "title": document.title,
        "text": document.text,
        "source_path": document.source_path,
        "aliases": document.aliases,
        "metadata": document.metadata,
    }
