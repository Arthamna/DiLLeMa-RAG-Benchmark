#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ENV="$REPO_ROOT/.venv-vllm-gpu"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv tidak ditemukan. Instal dari https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv python install 3.12
export UV_PROJECT_ENVIRONMENT="$VLLM_ENV"
uv sync --project "$REPO_ROOT/vllm_gpu" --python 3.12 --locked

"$VLLM_ENV/bin/python" - <<'PY'
import importlib.metadata

import torch
import vllm
from vllm import LLM

if importlib.metadata.version("vllm") != "0.32.0":
    raise SystemExit("Environment harus memakai vllm==0.32.0.")
if not torch.cuda.is_available():
    raise SystemExit("CUDA tidak terdeteksi oleh environment vLLM GPU.")
if not (torch.version.cuda or "").startswith("13.0"):
    raise SystemExit(f"vLLM 0.32.0 memerlukan runtime CUDA 13.0; ditemukan {torch.version.cuda}.")

capability = torch.cuda.get_device_capability(0)
architecture = f"sm_{capability[0]}{capability[1]}"
architectures = torch.cuda.get_arch_list()
if architecture not in architectures:
    raise SystemExit(
        f"GPU memerlukan {architecture}, tetapi wheel PyTorch hanya mendukung {architectures}."
    )

print("vLLM:", vllm.__version__)
print("vLLM LLM class:", LLM)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", capability)
print("Wheel architectures:", architectures)
print("CUDA smoke test:", torch.ones(1, device="cuda"))
PY
