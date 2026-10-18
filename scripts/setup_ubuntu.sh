#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 tidak ditemukan. Instal python3 dan python3-venv terlebih dahulu." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  echo "Membuat virtual environment lokal: $REPO_ROOT/.venv"
else
  echo "Menggunakan virtual environment yang sudah ada: $REPO_ROOT/.venv"
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_QUANTIZED:-1}" == "1" ]]; then
  python -m pip install -e ".[dev,rrag,quantized]"
else
  python -m pip install -e ".[dev,rrag]"
fi

python -c 'import torch; print("CUDA tersedia:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "tidak terdeteksi; pipeline akan memakai CPU")'
echo "Setup selesai. Aktifkan kembali dengan: source .venv/bin/activate"
