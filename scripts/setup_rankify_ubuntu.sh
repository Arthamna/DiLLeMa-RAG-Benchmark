#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

uv python install
export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv-rankify"
uv sync --project "$REPO_ROOT/src/rag_benchmark/techniques/rankify/worker" \
  --python 3.12 --extra legacy --locked --reinstall-package dillema-rankify-worker
"$JAVA_HOME/bin/javac" -version
"$REPO_ROOT/.venv-rankify/bin/python" -m rag_benchmark_rankify.worker <<<'{"schema_version":1,"operation":"capabilities"}'
