from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TechniqueSpec:
    name: str
    import_path: str


TECHNIQUES = {
    "dillema": TechniqueSpec("dillema", "rag_benchmark.techniques.dillema.pipeline:RAGPipeline"),
}


def get_technique(name: str) -> TechniqueSpec:
    try:
        return TECHNIQUES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported RAG technique: {name}") from exc

