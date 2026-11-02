from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TechniqueSpec:
    name: str
    import_path: str


TECHNIQUES = {
    "dillema": TechniqueSpec("dillema", "rag_benchmark.techniques.dillema.pipeline:RAGPipeline"),
    "qwen3_embedding": TechniqueSpec("qwen3_embedding", "rag_benchmark.techniques.qwen3.pipeline:Qwen3Pipeline"),
    "qwen3_embedding_rerank": TechniqueSpec("qwen3_embedding_rerank", "rag_benchmark.techniques.qwen3.pipeline:Qwen3Pipeline"),
    "bge_m3_dense": TechniqueSpec("bge_m3_dense", "rag_benchmark.techniques.bge_m3.pipeline:BgeM3Pipeline"),
    "bge_m3_sparse": TechniqueSpec("bge_m3_sparse", "rag_benchmark.techniques.bge_m3.pipeline:BgeM3Pipeline"),
    "bge_m3_multi_vector": TechniqueSpec("bge_m3_multi_vector", "rag_benchmark.techniques.bge_m3.pipeline:BgeM3Pipeline"),
    "bge_m3_dense_sparse": TechniqueSpec("bge_m3_dense_sparse", "rag_benchmark.techniques.bge_m3.pipeline:BgeM3Pipeline"),
    "bge_m3_all": TechniqueSpec("bge_m3_all", "rag_benchmark.techniques.bge_m3.pipeline:BgeM3Pipeline"),
    "self_rag_adaptive": TechniqueSpec("self_rag_adaptive", "rag_benchmark.techniques.self_rag.pipeline:SelfRagPipeline"),
    "self_rag_always": TechniqueSpec("self_rag_always", "rag_benchmark.techniques.self_rag.pipeline:SelfRagPipeline"),
}


def get_technique(name: str) -> TechniqueSpec:
    try:
        return TECHNIQUES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported RAG technique: {name}") from exc
