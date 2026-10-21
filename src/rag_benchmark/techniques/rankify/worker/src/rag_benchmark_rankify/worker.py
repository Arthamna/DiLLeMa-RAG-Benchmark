from __future__ import annotations

import copy
import importlib
import importlib.metadata
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from .config import PRESETS, _configure_java_home, _device
from .generation import _custom_prompt, _generation_kwargs, _generator
from .indexing import _index
from .retrieval import _context_values, _document, _reranker, _retrieve, _retrievers

LOGGER = logging.getLogger(__name__)

PRESETS = {
    "rankify_bm25_basic": ("bm25", False, "basic-rag"),
    "rankify_bge_basic": ("bge", False, "basic-rag"),
    "rankify_bm25_bge_rerank_basic": ("bm25", True, "basic-rag"),
    "rankify_bge_bge_rerank_basic": ("bge", True, "basic-rag"),
    "rankify_hybrid_bge_rerank_basic": ("hybrid", True, "basic-rag"),
    "rankify_hybrid_bge_rerank_cot": ("hybrid", True, "chain-of-thought-rag"),
    "rankify_hybrid_bge_rerank_incontext": ("hybrid", True, "in-context-ralm"),
}


def _ensure_rankify_vllm_import() -> bool:
    """Provide Rankify's unused eager vLLM import without installing vLLM here."""
    try:
        module = importlib.import_module("vllm")
    except ModuleNotFoundError as error:
        if error.name != "vllm":
            raise

        class UnavailableLLM:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "Rankify's in-process vLLM backend is disabled in this environment. "
                    "Run vLLM from .venv-vllm-gpu and use generator.backend="
                    "openai-compatible instead."
                )

        module = type(sys)("vllm")
        module.LLM = UnavailableLLM
        module.__dillema_import_shim__ = True
        sys.modules["vllm"] = module
    return bool(getattr(module, "__dillema_import_shim__", False))


def _configure_java_home() -> str:
    configured = os.environ.get("JAVA_HOME")
    if configured:
        java_bin = Path(configured) / "bin"
        if any((java_bin / name).is_file() for name in ("javac", "javac.exe")):
            return configured

    javac = shutil.which("javac")
    if not javac:
        raise RuntimeError(
            "Rankify BM25 requires JDK 21 (javac), not only a JRE. "
            "On Ubuntu run: sudo apt install openjdk-21-jdk-headless"
        )
    java_home = str(Path(javac).resolve().parent.parent)
    os.environ["JAVA_HOME"] = java_home
    return java_home


def _device(request: dict) -> tuple[str, bool]:
    requested = str(request.get("device") or "auto").casefold()
    try:
        import torch

        available = bool(torch.cuda.is_available())
    except ImportError:
        available = False
    if requested == "cpu":
        return "cpu", False
    if requested.startswith("cuda") and not available:
        LOGGER.warning("CUDA diminta tetapi tidak tersedia; Rankify memakai CPU.")
        return "cpu", True
    if requested == "auto":
        if available:
            return "cuda", False
        LOGGER.warning("GPU CUDA tidak terdeteksi; evaluasi Rankify memakai CPU.")
        return "cpu", True
    return requested, False


def _capabilities() -> dict:
    import numpy
    import pyarrow
    import torch
    vllm_shimmed = _ensure_rankify_vllm_import()
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
        "rankify_vllm_import": "shim" if vllm_shimmed else "installed",
        "presets": list(PRESETS),
    }


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


def _generator(request: dict, method: str, device: str):
    _ensure_rankify_vllm_import()
    from rankify.generator.generator import Generator

    config = request["generator"]
    factory_method = "basic-rag" if method == "in-context-ralm" else method
    common = {"method": factory_method, "model_name": str(config["model"])}
    if config["backend"] == "openai-compatible":
        key_env = str(config.get("api_key_env") or "LLM_API_KEY")
        common.update({
            "backend": "openai",
            "api_keys": [os.environ[key_env]],
            "base_url": str(config["base_url"]),
        })
    else:
        common.update({
            "backend": "huggingface",
            "trust_remote_code": bool(config.get("trust_remote_code", False)),
        })
        quantization = str(config.get("quantization") or "none").casefold()
        if device == "cuda":
            common["device_map"] = "auto"
            if quantization in {"4bit", "8bit"}:
                from transformers import BitsAndBytesConfig

                common["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=quantization == "4bit",
                    load_in_8bit=quantization == "8bit",
                )
        elif quantization != "none":
            LOGGER.warning("Quantization %s dinonaktifkan karena generator memakai CPU.", quantization)
    return Generator(**common)


def _generation_kwargs(request: dict) -> dict:
    maximum = int(request["generator"]["max_tokens"])
    if request["generator"]["backend"] == "openai-compatible":
        return {"temperature": 0, "max_tokens": maximum}
    return {"do_sample": False, "max_new_tokens": maximum}


def _custom_prompt(method: str):
    if method != "in-context-ralm":
        return None
    from rankify.generator.prompt_template import DEFAULT_PROMPTS, PromptTemplate

    return DEFAULT_PROMPTS[PromptTemplate.IN_CONTEXT_RALM]


def _context_values(document, top_k: int):
    reranked = getattr(document, "reorder_contexts", None)
    contexts = reranked if reranked else getattr(document, "contexts", [])
    return list(contexts)[:top_k]


def _metadata(corpus_path: str) -> dict[str, dict]:
    values = {}
    with Path(corpus_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                values[str(item["id"])] = item
    return values


def _evaluate_batch(request: dict, batch: dict, writer, mode: str, method: str, device: str, fallback: bool, ranker, generator):
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
                "id": question["id"], "language": question["language"], "question": question["question"],
                "prediction": "", "evidence": [], "context_length": 0, "status": "error", "error": None,
                "diagnostics": {
                    "gpu_fallback": fallback,
                    "device": device,
                    "rag_method": method,
                    "prompt_adapter": "basic-rag-with-native-in-context-ralm-prompt"
                    if method == "in-context-ralm" else None,
                },
            }
            if not question.get("available", True):
                row["status"] = "missing_context"
            else:
                try:
                    document = _document(question["question"])
                    retrieval_started = time.perf_counter()
                    document = _retrieve(document, retrievers, mode, int(request["rrf_k"]), int(request["candidates"]))
                    row["diagnostics"]["retrieval_latency_ms"] = round((time.perf_counter() - retrieval_started) * 1000, 3)
                    rerank_started = time.perf_counter()
                    if ranker is not None:
                        ranked = ranker.rank([document])
                        document = ranked[0] if ranked else document
                    row["diagnostics"]["rerank_latency_ms"] = round((time.perf_counter() - rerank_started) * 1000, 3)
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
                        [document], custom_prompt=_custom_prompt(method), **_generation_kwargs(request),
                    )
                    row["prediction"] = str(generated[0] if generated else "").strip()
                    row["diagnostics"]["generation_latency_ms"] = round((time.perf_counter() - generation_started) * 1000, 3)
                    row["status"] = "ok" if row["prediction"] else ("abstention" if evidence else "no_evidence")
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
        "status": "ok", "operation": "evaluate", "preset": preset,
        "device": device, "gpu_fallback": fallback, "output_path": str(output.resolve()),
        "dense_model": request["dense_model"], "reranker_model": request["reranker_model"],
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
