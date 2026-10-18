from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass

from ...shared.config import PipelineConfig
from ...shared.records import DocumentRecord, RetrievedEvidence, TechniqueResult
from ...shared.retrieval import DocumentChunker

_LISTING = re.compile(
    r"\bapa\s*(?:saja|aja)\b|\bsiapa\s*(?:saja|aja)\b|\bmana\s*(?:saja|aja)\b"
    r"|\bdaftar\b|\bsebutkan\b|\brinci(?:an)?\b|\b(?:semua|seluruh)\b"
    r"|\blist\s+(?:of|all)\b|\bwhat\s+are\s+(?:all\s+)?the\b|\blist\s+all\b"
    r"|\bquali\s+sono\b|\belenca\b|\belencare\b|\b(?:tutti|tutte)\b",
    re.IGNORECASE,
)


def is_listing(question: str) -> bool:
    return bool(_LISTING.search(question or ""))


@dataclass(slots=True)
class PackedContext:
    text: str
    evidence: list[RetrievedEvidence]
    diagnostics: dict | None = None


def drop_weak_evidence(items: list[RetrievedEvidence], ratio: float) -> list[RetrievedEvidence]:
    if not items or ratio <= 0:
        return items
    floor = max(item.rerank_score for item in items) * ratio
    return [item for item in items if item.rerank_score >= floor]


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


class RAGPipeline:
    name = "dillema"

    def __init__(self, store, embedder, reranker, config: PipelineConfig | None = None, generator=None):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or PipelineConfig()
        self.generator = generator

    def index(self, collection: str, documents: list[DocumentRecord]) -> int:
        chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_chars=self.config.min_chunk_chars,
            parent_window_chars=self.config.parent_window_chars,
        )
        chunks = chunker.chunk_documents(documents)
        self.store.recreate_collection(collection, self.embedder.vector_size)
        if chunks:
            vectors = self.embedder.encode_passages([chunk.text for chunk in chunks])
            self.store.add_chunks(collection, chunks, vectors)
        return len(chunks)

    def retrieve(self, collection: str, question: str) -> PackedContext:
        retrieval_started = time.perf_counter()
        query_vector = self.embedder.encode_query(question)
        hits = self.store.search(collection, query_vector, question, self.config.retrieval_candidates)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        payloads = [dict(hit.payload or {}) for hit in hits]
        passages = [str(payload.get("document") or "") for payload in payloads]
        rerank_started = time.perf_counter()
        scores = self.reranker.score(question, passages)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        ranked: list[RetrievedEvidence] = []
        for payload, passage, score in zip(payloads, passages, scores):
            if not passage or score < self.config.rerank_min_score:
                continue
            metadata = {
                **dict(payload.get("metadata") or {}),
                "parent_id": payload.get("parent_id"),
                "parent_window": payload.get("parent_window"),
                "evidence_text": payload.get("evidence_text"),
                "source_path": payload.get("source_path"),
                "title": payload.get("title"),
            }
            ranked.append(RetrievedEvidence(
                source_id=str(payload.get("document_id") or ""),
                chunk_id=str(payload.get("chunk_id") or ""),
                text=passage,
                rerank_score=float(score),
                metadata=metadata,
            ))
        ranked.sort(key=lambda item: item.rerank_score, reverse=True)
        listing = is_listing(question)
        top = self.config.listing_top_results if listing else self.config.normal_top_results
        ratio = self.config.listing_relative_floor if listing else self.config.rerank_relative_floor
        parents = self.config.listing_max_parents if listing else self.config.max_parents
        packed = pack_parent_pages(drop_weak_evidence(ranked[:top], ratio), parents)
        packed.diagnostics = {
            "retrieval_latency_ms": round(retrieval_ms, 3),
            "rerank_latency_ms": round(rerank_ms, 3),
            "retrieved_candidates": len(hits),
        }
        return packed

    def answer(self, collection: str, question: str) -> TechniqueResult:
        packed = self.retrieve(collection, question)
        generation_started = time.perf_counter()
        prediction = self.generator.generate(question, packed.text) if packed.text and self.generator else ""
        generation_ms = (time.perf_counter() - generation_started) * 1000
        return TechniqueResult(
            prediction=prediction,
            context=packed.text,
            evidence=packed.evidence,
            diagnostics={
                **(packed.diagnostics or {}),
                "generation_latency_ms": round(generation_ms, 3),
                "listing": is_listing(question),
                "selected_evidence": len(packed.evidence),
                "selected_parents": len({item.metadata.get("parent_id") or item.source_id for item in packed.evidence}),
            },
        )


DiLLeMaTechnique = RAGPipeline

