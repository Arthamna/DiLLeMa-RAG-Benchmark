from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TechniqueSpec:
    name: str
    import_path: str


TECHNIQUES = {
    "dillema": TechniqueSpec("dillema", "rag_benchmark.techniques.dillema.pipeline:RAGPipeline"),
    "retro": TechniqueSpec("retro", "rag_benchmark.techniques.retro.pipeline:RetroPipeline"),
    "piperag": TechniqueSpec("piperag", "rag_benchmark.techniques.piperag.pipeline:PipeRagPipeline"),
}

for _group, _pipeline, _variants in (
    ("bge_m3", "BgeM3Pipeline", ("dense", "sparse", "multi_vector", "dense_sparse", "all")),
    ("self_rag", "SelfRagPipeline", ("adaptive", "always")),
):
    for _variant in _variants:
        _name = f"{_group}_{_variant}"
        TECHNIQUES[_name] = TechniqueSpec(_name, f"rag_benchmark.techniques.{_group}.pipeline:{_pipeline}")


def get_technique(name: str) -> TechniqueSpec:
    try:
        return TECHNIQUES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported RAG technique: {name}") from exc
