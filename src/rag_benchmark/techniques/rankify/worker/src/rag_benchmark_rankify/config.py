from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

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
