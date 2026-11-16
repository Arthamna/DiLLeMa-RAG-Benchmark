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
  uv sync --locked --extra api --extra dev --extra hf --extra quantized --extra paper --extra inference
else
  uv sync --locked --extra api --extra dev --extra hf --extra paper --extra inference
fi

uv run --no-sync python - <<'PY'
import onnxruntime as ort
import torch

if not torch.cuda.is_available():
    print("CUDA tidak terdeteksi; pipeline akan memakai CPU.")
    raise SystemExit(0)

capability = torch.cuda.get_device_capability(0)
architecture = f"sm_{capability[0]}{capability[1]}"
architectures = torch.cuda.get_arch_list()
if not str(torch.version.cuda).startswith("13.2"):
    raise SystemExit(f"CUDA runtime PyTorch harus 13.2, ditemukan {torch.version.cuda}.")
if architecture not in architectures:
    raise SystemExit(
        f"GPU memerlukan {architecture}, tetapi wheel PyTorch hanya mendukung {architectures}."
    )
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit(f"ONNX Runtime tidak menyediakan CUDAExecutionProvider: {ort.get_available_providers()}")

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", capability)
print("Wheel architectures:", architectures)
print("ONNX Runtime:", ort.__version__)
print("ONNX providers:", ort.get_available_providers())
print("CUDA smoke test:", torch.ones(1, device="cuda"))
PY
echo "Setup selesai. Jalankan command project dengan: uv run --no-sync <command>"
echo "Jalankan service RETRO/PipeRAG dengan: docker compose up --build -d"
