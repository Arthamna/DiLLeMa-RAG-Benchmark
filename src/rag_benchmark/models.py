from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentRecord:
    id: str
    dataset: str
    language: str
    title: str
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QARecord:
    id: str
    dataset: str
    language: str
    question: str
    references: list[str]
    split: str
    available: bool = True
    source_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ChunkRecord:
    id: str
    document_id: str
    dataset: str
    language: str
    text: str
    evidence_text: str
    parent_window: str
    parent_id: str
    source_path: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedEvidence:
    source_id: str
    chunk_id: str
    text: str
    rerank_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PredictionRecord:
    id: str
    dataset: str
    language: str
    question: str
    references: list[str]
    prediction: str
    retrieved_source_ids: list[str]
    rerank_scores: list[float]
    context_length: int
    latency_ms: float
    status: str
    available: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

