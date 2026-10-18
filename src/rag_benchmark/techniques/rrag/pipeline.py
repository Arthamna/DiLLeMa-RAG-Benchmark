from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...shared.config import PipelineConfig
from ...shared.local_llm import rrag_prompt
from ...shared.records import DocumentRecord, RetrievedEvidence, TechniqueResult
from ...shared.retrieval import DocumentChunker
from .config import RRAGConfig
from .features import compute_retrieval_features


@dataclass(slots=True)
class RRAGContext:
    text: str
    evidence: list[RetrievedEvidence]
    features: np.ndarray


def evidence_matches_sources(evidence: RetrievedEvidence, source_ids: list[str]) -> bool:
    candidates = {
        evidence.source_id,
        str(evidence.metadata.get("source_path") or ""),
        Path(str(evidence.metadata.get("source_path") or "")).name,
        str(evidence.metadata.get("source_file") or ""),
    }
    candidates.update(str(item) for item in evidence.metadata.get("aliases", []))
    normalized = {candidate.casefold() for candidate in candidates if candidate}
    return any(str(source_id).casefold() in normalized for source_id in source_ids)


class RRAGTechnique:
    name = "rrag"
    supports_training = True

    def __init__(
        self, store, embedder, runtime=None, adapter=None,
        pipeline_config: PipelineConfig | None = None,
        rrag_config: RRAGConfig | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.runtime = runtime
        self.adapter = adapter
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.rrag_config = rrag_config or RRAGConfig()

    def index(self, collection: str, documents: list[DocumentRecord]) -> int:
        chunker = DocumentChunker(
            chunk_size=self.pipeline_config.chunk_size,
            chunk_overlap=self.pipeline_config.chunk_overlap,
            min_chunk_chars=self.pipeline_config.min_chunk_chars,
            parent_window_chars=self.pipeline_config.parent_window_chars,
        )
        chunks = chunker.chunk_documents(documents)
        self.store.recreate_collection(collection, self.embedder.vector_size)
        if chunks:
            vectors = self.embedder.encode_passages([chunk.text for chunk in chunks])
            self.store.add_chunks(collection, chunks, vectors)
        return len(chunks)

    def retrieve(self, collection: str, question: str) -> RRAGContext:
        query_vector = self.embedder.encode_query(question)
        hits = self.store.search(
            collection, query_vector, question, self.pipeline_config.retrieval_candidates,
        )
        payloads = [dict(hit.payload or {}) for hit in hits]
        passages = [str(payload.get("document") or "") for payload in payloads]
        valid = [(payload, passage) for payload, passage in zip(payloads, passages) if passage]
        if not valid:
            return RRAGContext("", [], np.empty((0, 3), dtype=np.float32))
        payloads, passages = map(list, zip(*valid))
        passage_vectors = self.embedder.encode_passages(passages)
        features, order = compute_retrieval_features(
            query_vector, passage_vectors, self.rrag_config.top_k,
        )
        evidence: list[RetrievedEvidence] = []
        for feature, source_index in zip(features, order):
            payload = payloads[int(source_index)]
            metadata = {
                **dict(payload.get("metadata") or {}),
                "source_path": payload.get("source_path"),
                "title": payload.get("title"),
                "retrieval_features": [float(value) for value in feature],
            }
            evidence.append(RetrievedEvidence(
                source_id=str(payload.get("document_id") or ""),
                chunk_id=str(payload.get("chunk_id") or ""),
                text=str(payload.get("document") or ""),
                rerank_score=float(feature[0]),
                metadata=metadata,
            ))
        context = "\n".join(
            f"[{index}] {item.metadata.get('title') or item.source_id}: {item.text}"
            for index, item in enumerate(evidence, 1)
        )
        return RRAGContext(context, evidence, features)

    def labels_for(self, context: RRAGContext, source_ids: list[str]) -> list[float]:
        return [1.0 if evidence_matches_sources(item, source_ids) else 0.0 for item in context.evidence]

    def answer(self, collection: str, question: str) -> TechniqueResult:
        if self.runtime is None or self.adapter is None:
            raise RuntimeError("RRAG generation requires a LocalLLM runtime and trained adapter")
        import torch

        context = self.retrieve(collection, question)
        if not context.evidence:
            return TechniqueResult("", "", [], {"retrieval_features": [], "relevance_probabilities": []})
        self.adapter.to(self.runtime.device)
        self.adapter.eval()
        feature_tensor = torch.tensor(context.features, device=self.runtime.device).unsqueeze(0)
        with torch.no_grad():
            injected, logits, _ = self.adapter(feature_tensor)
            prediction = self.runtime.generate_with_injections(
                rrag_prompt(question, context.evidence), injected,
            )
        return TechniqueResult(
            prediction=prediction,
            context=context.text,
            evidence=context.evidence,
            diagnostics={
                "retrieval_features": context.features.tolist(),
                "relevance_probabilities": torch.sigmoid(logits).squeeze(0).cpu().tolist(),
            },
        )

