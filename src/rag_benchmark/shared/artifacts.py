from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _technique_root(index_root: Path, technique: str, dataset: str) -> Path:
    return index_root / technique / dataset


def _index_manifest_path(index_root: Path, technique: str, dataset: str) -> Path:
    return _technique_root(index_root, technique, dataset) / "index-manifest.json"


def _languages(dataset: str) -> list[str]:
    return ["id"] if dataset == "its" else ["en", "it"]


def _corpus_fingerprint(documents) -> str:
    payload = [
        {"id": item.id, "language": item.language, "title": item.title, "text": item.text}
        for item in sorted(documents, key=lambda value: value.id)
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
