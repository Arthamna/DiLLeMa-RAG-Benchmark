from __future__ import annotations

from pathlib import Path

from .config import _device


def _index(request: dict) -> dict:
    from rankify.indexing.bge_indexer import BGEIndexer
    from rankify.indexing.lucene_indexer import LuceneIndexer

    device, fallback = _device(request)
    corpus = Path(request["corpus_path"]).resolve()
    root = Path(request["index_dir"]).resolve()
    bm25_root = root / "bm25"
    bge_root = root / "bge"
    bm25 = LuceneIndexer(
        corpus_path=str(corpus), output_dir=str(bm25_root), index_type="custom", retriever_name="bm25",
    )
    bm25.build_index()
    bge_index = bge_root / "bge_index_custom"
    bge = BGEIndexer(
        corpus_path=str(corpus), encoder_name=str(request["dense_model"]),
        output_dir=str(bge_root), index_dir=str(bge_index), index_type="custom",
        retriever_name="bge", device=device,
    )
    bge.build_index()
    return {
        "status": "ok",
        "operation": "index",
        "device": device,
        "gpu_fallback": fallback,
        "corpus_path": str(corpus),
        "bm25_index": str(bm25.index_dir),
        "bge_index": str(bge_index),
        "dense_model": str(request["dense_model"]),
    }
