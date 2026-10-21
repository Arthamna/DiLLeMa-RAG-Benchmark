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
  uv sync --extra api --extra dev --extra hf --extra quantized --locked
else
  uv sync --extra api --extra dev --extra hf --locked
fi

uv run --no-sync python - <<'PY'
import torch

if not torch.cuda.is_available():
    print("CUDA tidak terdeteksi; pipeline akan memakai CPU.")
    raise SystemExit(0)

capability = torch.cuda.get_device_capability(0)
architecture = f"sm_{capability[0]}{capability[1]}"
architectures = torch.cuda.get_arch_list()
if not str(torch.version.cuda).startswith("13.2"):
    raise SystemExit(f"CUDA runtime harus 13.2, ditemukan {torch.version.cuda}.")
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
echo "Setup selesai. Jalankan command project dengan: uv run --no-sync <command>"
echo "Untuk matriks Rankify GPU, jalankan juga: bash scripts/setup_rankify_gpu_ubuntu.sh"
