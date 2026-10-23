from __future__ import annotations

import importlib.metadata
import json
import logging
import sys
import time
from pathlib import Path

from .config import PRESETS, _configure_java_home, _device
from .generation import _custom_prompt, _generation_kwargs, _generator
from .indexing import _index
from .retrieval import _context_values, _document, _reranker, _retrieve, _retrievers

LOGGER = logging.getLogger(__name__)


def _capabilities() -> dict:
    import numpy
    import pyarrow
    import torch
    from rankify.generator.generator import Generator
    from rankify.models.transformer_ranker import TransformerRanker
    from rankify.retrievers.bge_retriever import BGERetriever
    from rankify.retrievers.bm25_retriever import BM25Retriever

    version = importlib.metadata.version("rankify")
    if version != "0.1.4":
        raise RuntimeError(f"Expected rankify==0.1.4, found {version}")
    return {
        "status": "ok",
        "python_version": sys.version.split()[0],
        "rankify_version": version,
        "numpy_version": numpy.__version__,
        "pyarrow_version": pyarrow.__version__,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "retrievers": ["bm25", "bge"],
        "retriever_imports": [BM25Retriever.__name__, BGERetriever.__name__],
        "reranker_import": TransformerRanker.__name__,
        "generator_import": Generator.__name__,
        "presets": list(PRESETS),
    }


def _metadata(corpus_path: str) -> dict[str, dict]:
    values = {}
    with Path(corpus_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                values[str(item["id"])] = item
    return values


def _evaluate_batch(
    request: dict,
    batch: dict,
    writer,
    mode: str,
    method: str,
    device: str,
    fallback: bool,
    ranker,
    generator,
):
    retrievers = _retrievers(request, batch, mode, device)
    metadata = _metadata(batch["corpus_path"])
    top_k = int(request["top_k"])
    with Path(batch["questions_path"]).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            question = json.loads(line)
            started = time.perf_counter()
            row = {
                "id": question["id"],
                "language": question["language"],
                "question": question["question"],
                "prediction": "",
                "evidence": [],
                "context_length": 0,
                "status": "error",
                "error": None,
                "diagnostics": {
                    "gpu_fallback": fallback,
                    "device": device,
                    "rag_method": method,
                    "prompt_adapter": (
                        "basic-rag-with-native-in-context-ralm-prompt"
                        if method == "in-context-ralm"
                        else None
                    ),
                },
            }
            if not question.get("available", True):
                row["status"] = "missing_context"
            else:
                try:
                    document = _document(question["question"])
                    retrieval_started = time.perf_counter()
                    document = _retrieve(
                        document,
                        retrievers,
                        mode,
                        int(request["rrf_k"]),
                        int(request["candidates"]),
                    )
                    row["diagnostics"]["retrieval_latency_ms"] = round(
                        (time.perf_counter() - retrieval_started) * 1000, 3
                    )
                    rerank_started = time.perf_counter()
                    if ranker is not None:
                        ranked = ranker.rank([document])
                        document = ranked[0] if ranked else document
                    row["diagnostics"]["rerank_latency_ms"] = round(
                        (time.perf_counter() - rerank_started) * 1000, 3
                    )
                    contexts = _context_values(document, top_k)
                    document.contexts = contexts
                    if hasattr(document, "reorder_contexts"):
                        document.reorder_contexts = contexts
                    evidence = []
                    for context in contexts:
                        item = metadata.get(str(context.id), {})
                        evidence.append({
                            "source_id": str(item.get("source_id") or context.id),
                            "chunk_id": str(context.id),
                            "text": str(context.text or ""),
                            "score": float(context.score or 0.0),
                            "metadata": {
                                **dict(item.get("metadata") or {}),
                                "title": item.get("title") or getattr(context, "title", ""),
                                "source_path": item.get("source_path") or "",
                            },
                        })
                    context_text = "\n\n".join(item["text"] for item in evidence)
                    row["evidence"] = evidence
                    row["context_length"] = len(context_text)
                    generation_started = time.perf_counter()
                    generated = generator.generate(
                        [document],
                        custom_prompt=_custom_prompt(method),
                        **_generation_kwargs(request),
                    )
                    row["prediction"] = str(generated[0] if generated else "").strip()
                    row["diagnostics"]["generation_latency_ms"] = round(
                        (time.perf_counter() - generation_started) * 1000, 3
                    )
                    row["status"] = (
                        "ok" if row["prediction"] else ("abstention" if evidence else "no_evidence")
                    )
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    LOGGER.warning("Rankify QA failed %s: %s", question["id"], row["error"])
            row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def _evaluate(request: dict) -> dict:
    preset = str(request["preset"])
    if preset not in PRESETS:
        raise ValueError(f"Unsupported Rankify preset: {preset}")
    mode, rerank, method = PRESETS[preset]
    device, fallback = _device(request)
    ranker = _reranker(request, rerank, device)
    generator = _generator(request, method, device)
    output = Path(request["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as writer:
        for batch in request["batches"]:
            _evaluate_batch(request, batch, writer, mode, method, device, fallback, ranker, generator)
    return {
        "status": "ok",
        "operation": "evaluate",
        "preset": preset,
        "device": device,
        "gpu_fallback": fallback,
        "output_path": str(output.resolve()),
        "dense_model": request["dense_model"],
        "reranker_model": request["reranker_model"],
        "generator_model": request["generator"]["model"],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    try:
        request = json.loads(sys.stdin.readline())
        if int(request.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported worker schema_version")
        java_home = _configure_java_home()
        operation = request.get("operation")
        if operation == "capabilities":
            response = _capabilities()
            response["java_home"] = java_home
        elif operation == "index":
            response = _index(request)
        elif operation == "evaluate":
            response = _evaluate(request)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    except Exception as exc:
        LOGGER.exception("Rankify worker failed")
        response = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
