from __future__ import annotations

import copy
from pathlib import Path


def _document(question: str):
    from rankify.dataset.dataset import Answer, Document, Question

    return Document(question=Question(question), answers=Answer([""]))


def _retrievers(request: dict, batch: dict, mode: str, device: str):
    from rankify.retrievers.bge_retriever import BGERetriever
    from rankify.retrievers.bm25_retriever import BM25Retriever

    root = Path(batch["index_dir"])
    values = {}
    if mode in {"bm25", "hybrid"}:
        values["bm25"] = BM25Retriever(
            n_docs=int(request["candidates"]),
            index_type="custom",
            index_folder=str(root / "bm25"),
        )
    if mode in {"bge", "hybrid"}:
        values["bge"] = BGERetriever(
            n_docs=int(request["candidates"]),
            index_folder=str(root / "bge" / "bge_index_custom"),
            model=str(request["dense_model"]), device=device,
        )
    return values


def _rrf(left, right, k: int, limit: int):
    contexts = {}
    scores = {}
    for ranking in (left, right):
        for rank, context in enumerate(ranking, 1):
            identifier = str(context.id)
            contexts.setdefault(identifier, context)
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
    result = []
    for identifier in ordered:
        context = contexts[identifier]
        context.score = float(scores[identifier])
        result.append(context)
    return result


def _retrieve(document, retrievers: dict, mode: str, rrf_k: int, candidates: int):
    if mode != "hybrid":
        return retrievers[mode].retrieve([document])[0]
    sparse = retrievers["bm25"].retrieve([copy.deepcopy(document)])[0]
    dense = retrievers["bge"].retrieve([copy.deepcopy(document)])[0]
    document.contexts = _rrf(sparse.contexts, dense.contexts, rrf_k, candidates)
    return document


def _reranker(request: dict, enabled: bool, device: str):
    if not enabled:
        return None
    from rankify.models.transformer_ranker import TransformerRanker

    configured = str(request["reranker_model"])
    return TransformerRanker(method="transformer_ranker", model_name=configured, device=device)


def _context_values(document, top_k: int):
    reranked = getattr(document, "reorder_contexts", None)
    contexts = reranked if reranked else getattr(document, "contexts", [])
    return list(contexts)[:top_k]
