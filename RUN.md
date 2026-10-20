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
  --project "$REPO_ROOT/rankify_worker" \
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
