# Menjalankan benchmark

Keempat preset kini juga menjalankan `qwen3_embedding`,
`qwen3_embedding_rerank`, dan lima varian BGE-M3 (15 pipeline). Setelah update, jalankan
`bash scripts/setup_ubuntu.sh` untuk constraint Transformers >=4.51; pada RTX 5080
gunakan `bash scripts/setup_rankify_gpu_ubuntu.sh`. Jalankan `--stage all` sekali
untuk budget baru top-100. Generator tetap mengikuti `.env` dan YAML sebelumnya.
Setup utama memasang extra `bge-m3`; konfigurasi CUDA 13.2, PyTorch 2.13, dan
environment GPU terisolasi tetap mengikuti setup RTX 50. Detail lifecycle GPU,
opsi 4-bit, output Qwen3, dan arsitektur hybrid BGE-M3 ada di README.

Implementasi teknik ada di `src/rag_benchmark/techniques/dillema/` dan
`src/rag_benchmark/techniques/rankify/`. Komponen bersama ada di
`src/rag_benchmark/shared/`. Proyek worker beserta `pyproject.toml` dan `uv.lock`
sekarang ada di `src/rag_benchmark/techniques/rankify/worker/`.
Setelah menarik refactor ini, jalankan ulang setup Rankify agar instalasi
editable menunjuk lokasi source baru. Environment GPU tetap `.venv-rankify-gpu`;
command worker dan konfigurasi YAML tetap sama.

bash scripts/setup_ubuntu.sh
bash scripts/setup_rankify_gpu_ubuntu.sh

sync :

REPO_ROOT="$(pwd -P)"
UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv" \
uv sync \
  --project "$REPO_ROOT" \
  --python 3.12 \
  --extra api \
  --extra dev \
  --extra hf \
  --extra quantized \
  --extra bge-m3


"$REPO_ROOT/.venv/bin/python" -c \
  'import sys, rag_benchmark; print(sys.prefix); print(rag_benchmark.__file__)'


UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv-rankify-gpu" \
uv sync \
  --project "$REPO_ROOT/src/rag_benchmark/techniques/rankify/worker" \
  --python 3.12 \
  --locked

"$REPO_ROOT/.venv-rankify-gpu/bin/python" -c \
  'import sys, importlib.metadata as m, rag_benchmark_rankify, rankify; print(sys.prefix); print(rag_benchmark_rankify.__file__); print(m.version("dillema-rankify-worker")); print(m.version("rankify"))'

"$REPO_ROOT/.venv/bin/python" -m rag_benchmark benchmark \
  --config configs/experiments/rankify-multilingual.yaml \
  --stage all \
  --output artifacts/benchmarks/its-run

# Smoke test BGE-M3 gabungan saja. `bge_m3_all` adalah satu ranking hybrid,
# bukan alias untuk menjalankan kelima varian.
"$REPO_ROOT/.venv/bin/python" -m rag_benchmark index \
  --config configs/experiments/rankify-multilingual.yaml \
  --technique bge_m3_all \
  --output artifacts/benchmarks/its-bge-m3-all

"$REPO_ROOT/.venv/bin/python" -m rag_benchmark evaluate \
  --config configs/experiments/rankify-multilingual.yaml \
  --technique bge_m3_all \
  --output artifacts/benchmarks/its-bge-m3-all

# Benchmark penuh menghasilkan bge_m3_dense/, bge_m3_sparse/,
# bge_m3_multi_vector/, bge_m3_dense_sparse/, bge_m3_all/, serta
# bge-m3-ablation.json di direktori output.

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
