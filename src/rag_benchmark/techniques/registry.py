from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TechniqueSpec:
    name: str
    supports_training: bool
    import_path: str


TECHNIQUES = {
    "dillema": TechniqueSpec("dillema", False, "rag_benchmark.techniques.dillema.pipeline:RAGPipeline"),
    "rrag": TechniqueSpec("rrag", True, "rag_benchmark.techniques.rrag.pipeline:RRAGTechnique"),
}


def get_technique(name: str) -> TechniqueSpec:
    try:
        return TECHNIQUES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported RAG technique: {name}") from exc

