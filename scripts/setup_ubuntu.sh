#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv tidak ditemukan. Instal dari https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv python install

if [[ "${INSTALL_QUANTIZED:-1}" == "1" ]]; then
  uv sync --extra api --extra dev --extra hf --extra quantized
else
  uv sync --extra api --extra dev --extra hf
fi

uv run --no-sync python -c 'import torch; print("CUDA tersedia:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "tidak terdeteksi; pipeline akan memakai CPU")'
echo "Setup selesai. Jalankan command project dengan: uv run --no-sync <command>"
echo "Untuk matriks Rankify, jalankan juga: bash scripts/setup_rankify_ubuntu.sh"
