#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv tidak ditemukan. Instal dari https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv python install
export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv-rankify"
uv sync --project "$REPO_ROOT/rankify_worker"
"$REPO_ROOT/.venv-rankify/bin/python" -m rag_benchmark_rankify.worker <<<'{"schema_version":1,"operation":"capabilities"}'
