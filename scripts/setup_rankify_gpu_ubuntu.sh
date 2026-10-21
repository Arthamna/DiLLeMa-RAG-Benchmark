#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ENV="$REPO_ROOT/.venv-rankify-gpu"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv tidak ditemukan. Instal dari https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if ! command -v javac >/dev/null 2>&1; then
  echo "JDK 21 tidak ditemukan. Instal dengan: sudo apt install openjdk-21-jdk-headless" >&2
  exit 1
fi
export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")"

uv python install 3.12
export UV_PROJECT_ENVIRONMENT="$GPU_ENV"
uv sync --project "$REPO_ROOT/rankify_worker_gpu" --python 3.12 --locked \
  --reinstall-package dillema-rankify-worker

"$GPU_ENV/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA tidak terdeteksi oleh environment Rankify GPU.")

capability = torch.cuda.get_device_capability(0)
architecture = f"sm_{capability[0]}{capability[1]}"
architectures = torch.cuda.get_arch_list()
if architecture not in architectures:
    raise SystemExit(
        f"GPU memerlukan {architecture}, tetapi wheel PyTorch hanya mendukung {architectures}."
    )

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", capability)
print("Wheel architectures:", architectures)
print("CUDA smoke test:", torch.ones(1, device="cuda"))
PY

"$JAVA_HOME/bin/javac" -version
"$GPU_ENV/bin/python" -m rag_benchmark_rankify.worker <<<'{"schema_version":1,"operation":"capabilities"}'
