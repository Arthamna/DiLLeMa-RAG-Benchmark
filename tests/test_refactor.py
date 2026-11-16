"""Regression checks for the retained DiLLeMa implementation and boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

from rag_benchmark.techniques.dillema.augmentation import pack_parent_pages
from rag_benchmark.techniques.dillema.generation import base_prompt
from rag_benchmark.shared.records import RetrievedEvidence

ROOT = Path(__file__).resolve().parents[1]


def test_dillema_prompt_and_parent_packing_remain_stable():
    packed = pack_parent_pages([
        RetrievedEvidence("a", "1", "bukti pertama", .9, {"parent_id": "a", "parent_window": "bukti pertama dan konteks"}),
        RetrievedEvidence("b", "2", "lainnya", .8, {"parent_id": "b"}),
        RetrievedEvidence("a", "3", "bukti terakhir", .7, {"parent_id": "a"}),
    ], 1)
    assert "bukti pertama dan konteks" in packed.text
    assert "bukti terakhir" in packed.text
    assert [item.chunk_id for item in packed.evidence] == ["1", "3"]
    assert "Answer using only the supplied evidence." in base_prompt("Apa aturannya?", "Bukti")


def test_runtime_never_imports_reference_checkout_or_removed_techniques():
    forbidden = {"rankify", "rank_zephyr", "qwen3", "bge_m3", "self_rag", "reference"}
    for path in (ROOT / "src" / "rag_benchmark").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(part in forbidden for name in names for part in name.split(".")), (path, names)


def test_only_expected_technique_directories_exist():
    names = {path.name for path in (ROOT / "src" / "rag_benchmark" / "techniques").iterdir() if path.is_dir() and path.name != "__pycache__"}
    assert names == {"dillema", "retro", "piperag"}


def test_inference_image_installs_project_with_uv_and_checks_import():
    dockerfile = (ROOT / "docker" / "Dockerfile.inference").read_text(encoding="utf-8")
    assert "nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04" in dockerfile
    assert 'uv pip install --python "$VIRTUAL_ENV/bin/python" --no-cache --no-deps .' in dockerfile
    assert "import rag_benchmark.techniques.retro.inference_service" in dockerfile
    assert "--system" not in dockerfile
