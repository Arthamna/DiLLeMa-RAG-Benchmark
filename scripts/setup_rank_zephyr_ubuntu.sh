#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "RankZephyr requires Linux x86_64 with an NVIDIA CUDA GPU." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

TECHNIQUE="$REPO_ROOT/src/rag_benchmark/techniques/rank_zephyr"
export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv-rank-llm"
uv sync --project "$TECHNIQUE/worker_env" --python 3.12 --locked
"$UV_PROJECT_ENVIRONMENT/bin/python" -I "$TECHNIQUE/worker.py" --probe
