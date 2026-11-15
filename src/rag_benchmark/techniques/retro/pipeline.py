from __future__ import annotations

from typing import Any

from ...shared.records import RetrievedEvidence, TechniqueResult


class RetroPipeline:
    name = "retro"

    def __init__(self, client, indexes: dict[str, str], generation: dict[str, Any]):
        self.client = client
        self.indexes = indexes
        self.generation = dict(generation)

    def answer(self, collection: str, question: str) -> TechniqueResult:
        if collection not in self.indexes:
            raise KeyError(f"No RETRO index for collection {collection}")
        response = self.client.generate({
            "mode": self.name,
            "index_id": self.indexes[collection],
            "question": question,
            **self.generation,
        })
        evidence = [RetrievedEvidence(
            source_id=str(item.get("source_id") or ""),
            chunk_id=str(item.get("chunk_id") or ""),
            text=str(item.get("text") or ""),
            rerank_score=float(item.get("score", 0.0)),
            metadata=dict(item.get("metadata") or {}),
        ) for item in response.get("evidence", [])]
        return TechniqueResult(
            prediction=str(response.get("prediction") or ""),
            context="\n\n".join(item.text for item in evidence),
            evidence=evidence,
            diagnostics=dict(response.get("diagnostics") or {}),
        )
