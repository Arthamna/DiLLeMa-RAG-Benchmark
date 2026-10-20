#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_ENV="$REPO_ROOT/.venv-vllm-gpu"

if [[ ! -x "$VLLM_ENV/bin/vllm" ]]; then
  echo "Environment vLLM belum tersedia. Jalankan scripts/setup_vllm_gpu_ubuntu.sh." >&2
  exit 1
fi
if [[ -z "${LLM_MODEL:-}" ]]; then
  echo "LLM_MODEL wajib diisi dengan model Hugging Face atau path model lokal." >&2
  exit 1
fi
if [[ -z "${LLM_API_KEY:-}" ]]; then
  echo "LLM_API_KEY wajib diisi sebagai token untuk endpoint lokal." >&2
  exit 1
fi

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"

echo "Menjalankan vLLM untuk $LLM_MODEL di http://$VLLM_HOST:$VLLM_PORT/v1"
exec "$VLLM_ENV/bin/vllm" serve "$LLM_MODEL" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --api-key "$LLM_API_KEY" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
