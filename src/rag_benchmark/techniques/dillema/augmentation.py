from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from ...shared.records import RetrievedEvidence


@dataclass(slots=True)
class PackedContext:
    text: str
    evidence: list[RetrievedEvidence]
    diagnostics: dict | None = None


def pack_parent_pages(items: list[RetrievedEvidence], max_parents: int) -> PackedContext:
    groups: OrderedDict[str, list[RetrievedEvidence]] = OrderedDict()
    for item in items:
        key = str(item.metadata.get("parent_id") or item.source_id)
        if key not in groups and len(groups) >= max_parents:
            continue
        groups.setdefault(key, []).append(item)

    bodies: list[str] = []
    kept: list[RetrievedEvidence] = []
    for group in groups.values():
        leaves: list[str] = []
        windows: list[str] = []
        for item in group:
            leaf = str(item.metadata.get("evidence_text") or item.text).strip()
            if leaf and leaf not in leaves:
                leaves.append(leaf)
            window = str(item.metadata.get("parent_window") or "").strip()
            if window and window not in windows:
                windows.append(window)
            kept.append(item)
        parts = list(leaves)
        for window in windows:
            candidate = [part for part in parts if part not in window] + [window]
            if len("\n\n".join(candidate)) <= 5000:
                parts = candidate
        source = group[0]
        label = str(source.metadata.get("title") or source.metadata.get("source_path") or source.source_id)
        bodies.append(f"Source: {label}\n" + "\n\n".join(parts))
    return PackedContext(text="\n\n---\n\n".join(bodies), evidence=kept, diagnostics={})
