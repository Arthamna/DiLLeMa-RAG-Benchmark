#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 tidak ditemukan. Instal python3 dan python3-venv terlebih dahulu." >&2
  exit 1
fi

if [[ ! -d .venv-rankify ]]; then
  python3 -m venv .venv-rankify
  echo "Membuat environment Rankify: $REPO_ROOT/.venv-rankify"
else
  echo "Menggunakan environment Rankify: $REPO_ROOT/.venv-rankify"
fi

source .venv-rankify/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ./rankify_worker
python -m rag_benchmark_rankify.worker <<<'{"schema_version":1,"operation":"capabilities"}'
