from __future__ import annotations

import os
from typing import Any


RANKIFY_PRESETS = (
    "rankify_bm25_basic",
    "rankify_bge_basic",
    "rankify_bm25_bge_rerank_basic",
    "rankify_bge_bge_rerank_basic",
    "rankify_hybrid_bge_rerank_basic",
    "rankify_hybrid_bge_rerank_cot",
    "rankify_hybrid_bge_rerank_incontext",
)


def _worker_common(spec) -> dict[str, Any]:
    return {
        "dense_model": spec.dense_model,
        "reranker_model": spec.reranker_model,
        "candidates": spec.candidates,
        "top_k": spec.top_k,
        "rrf_k": spec.rrf_k,
        "generator": spec.generator.public_dict(),
        "device": os.getenv("RAG_DEVICE", "auto"),
    }
