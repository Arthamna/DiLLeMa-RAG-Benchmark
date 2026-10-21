# Menjalankan benchmark

Keempat preset kini juga menjalankan `qwen3_embedding` dan
`qwen3_embedding_rerank` (sepuluh pipeline). Setelah update, jalankan
`bash scripts/setup_ubuntu.sh` untuk constraint Transformers >=4.51; pada RTX 5080
gunakan `bash scripts/setup_rankify_gpu_ubuntu.sh`. Jalankan `--stage all` sekali
untuk budget baru top-100. Generator tetap mengikuti `.env` dan YAML sebelumnya.
Detail lifecycle GPU, opsi 4-bit, dan output Qwen3 ada di README bagian Qwen3.

Implementasi teknik ada di `src/rag_benchmark/techniques/dillema/` dan
`src/rag_benchmark/techniques/rankify/`. Komponen bersama ada di
`src/rag_benchmark/shared/`. Proyek worker beserta `pyproject.toml` dan `uv.lock`
sekarang ada di `src/rag_benchmark/techniques/rankify/worker/`.
Setelah menarik refactor ini, jalankan ulang setup Rankify agar instalasi
editable menunjuk lokasi source baru. Environment tetap `.venv-rankify`;
command worker dan konfigurasi YAML tetap sama.

bash scripts/setup_ubuntu.sh
bash scripts/setup_rankify_ubuntu.sh

sync :

REPO_ROOT="$(pwd -P)"
UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv" \
uv sync \
  --project "$REPO_ROOT" \
  --python 3.12 \
  --extra api \
  --extra dev \
  --extra hf \
  --extra quantized


"$REPO_ROOT/.venv/bin/python" -c \
  'import sys, rag_benchmark; print(sys.prefix); print(rag_benchmark.__file__)'


UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv-rankify" \
uv sync \
  --project "$REPO_ROOT/src/rag_benchmark/techniques/rankify/worker" \
  --python 3.12 \
  --locked

"$REPO_ROOT/.venv-rankify/bin/python" -c \
  'import sys, importlib.metadata as m, rag_benchmark_rankify, rankify; print(sys.prefix); print(rag_benchmark_rankify.__file__); print(m.version("dillema-rankify-worker")); print(m.version("rankify"))'

"$REPO_ROOT/.venv/bin/python" -m rag_benchmark benchmark \
  --config configs/experiments/rankify-multilingual.yaml \
  --stage all \
  --output artifacts/benchmarks/its-run

"$REPO_ROOT/.venv/bin/python" -m rag_benchmark benchmark \
  --config configs/experiments/rankify-uniqa.yaml \
  --stage all \
  --output artifacts/benchmarks/uniqa-run

# LOCAL 

"$REPO_ROOT/.venv/bin/python" -m rag_benchmark benchmark \
  --config configs/experiments/rankify-multilingual-local.yaml \
  --stage all \
  --output artifacts/benchmarks/its-run

"$REPO_ROOT/.venv/bin/python" -m rag_benchmark benchmark \
  --config configs/experiments/rankify-multilingual-local.yaml \
  --stage all \
  --output artifacts/benchmarks/its-run

-- another notes :
- tokens and input tokens need to be adjustment
