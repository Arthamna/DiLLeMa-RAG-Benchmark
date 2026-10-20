# DiLLeMa–Rankify RAG Benchmark

Benchmark terkontrol untuk membandingkan full conventional RAG DiLLeMa dengan
tujuh pipeline Rankify pada corpus ITS dan UniQA. Indexing membangun retrieval
index; project ini tidak melakukan fine-tuning model.

Pipeline DiLLeMa:

```text
structured chunking → E5 dense + sparse Qdrant RRF
→ BGE reranking → relative filtering → parent packing → generation
```

Matriks Rankify mencakup BM25, BGE, hybrid RRF, BGE reranking, Basic RAG,
Chain-of-Thought RAG, dan In-Context RALM. RRAG/R²AG tidak lagi menjadi bagian
kode atau CLI. Artifact RRAG lama tidak dibaca dan dapat dihapus manual bila
sudah tidak diperlukan.

Rankify 0.1.4 membatasi kelas In-Context RALM ke model lokal dan tidak menerima
parameter decoding yang sama. Preset benchmark memakai generator Rankify
`basic-rag` dengan template native `IN_CONTEXT_RALM`, sehingga backend HF/API
tetap dapat dibandingkan dengan greedy decoding yang sama; adapter ini direkam
di diagnostics.

## Persyaratan Ubuntu

- Ubuntu 22.04 atau lebih baru.
- `uv` dan `curl` untuk mengelola Python serta dependency project.
- JDK 21 lengkap (`java` dan `javac`) untuk index BM25 Rankify/Pyserini.
- NVIDIA driver/CUDA opsional tetapi sangat disarankan.
- Ruang disk untuk model Hugging Face dan index.

```bash
sudo apt update
sudo apt install -y curl ca-certificates openjdk-21-jdk-headless
curl -LsSf https://astral.sh/uv/0.12.19/install.sh | sh
exec "$SHELL" -l
uv --version
```

## Setup environment lokal

Environment utama dan Rankify sengaja dipisahkan karena Rankify mengunci versi
beberapa dependency model.

```bash
cd /path/to/DiLLeMa-RAG-Benchmark
bash scripts/setup_ubuntu.sh
bash scripts/setup_rankify_gpu_ubuntu.sh
cp .env.example .env
```

Environment utama dan GPU Rankify dibuat oleh `uv` di `.venv` dan
`.venv-rankify-gpu`, tetap lokal di repository, dan diabaikan Git. Python
mengikuti `.python-version`; worker dipin pada `rankify==0.1.4`. Environment GPU
memakai `torch==2.13.0+cu132` dan `bitsandbytes==0.50.2`, termasuk kernel
Blackwell `sm_120` untuk RTX 50-series. Konfigurasi eksperimen bawaan menunjuk
ke `.venv-rankify-gpu/bin/python`.

Worker memanggil `TransformerRanker` secara langsung agar registry reranker
Rankify tidak memuat dependency LiT5/RankLLM seperti `fastchat` yang tidak dipakai.
Retriever BM25 dan BGE juga diimpor langsung agar registry retriever tidak
memuat `OnlineRetriever` dan dependency web-search yang tidak digunakan.
Compatibility worker dikunci pada PyArrow 17 dan Chardet 5.x.
Environment GPU tidak memasang extra reranking Rankify karena extra tersebut
menarik vLLM 0.7.0 dan PyTorch 2.5.1 yang tidak dipakai oleh direct
`TransformerRanker` serta tidak mendukung `sm_120`.

Untuk GPU lama atau fallback kompatibilitas, `.venv-rankify` tetap tersedia:

```bash
sudo apt install -y openjdk-21-jdk-headless
export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")"
java -version
javac -version

export UV_PROJECT_ENVIRONMENT="$PWD/.venv-rankify"
uv sync --project rankify_worker --python 3.12 --extra legacy --locked

printf '%s\n' '{"schema_version":1,"operation":"capabilities"}' \
  | .venv-rankify/bin/python -m rag_benchmark_rankify.worker
```

Worker mendeteksi JDK dari `javac` dan mengatur `JAVA_HOME` untuk subprocess-nya.
Probe yang berhasil berakhir dengan `"status": "ok"` dan menampilkan versi
Python, NumPy, PyArrow, Torch, lokasi Java, serta status CUDA.
Jika memakai environment legacy, ubah `rankify_python` pada YAML eksperimen
kembali menjadi `../../.venv-rankify/bin/python`.

## Konfigurasi generator

Salin `.env.example`, lalu pilih satu backend pada YAML eksperimen.

Hugging Face lokal:

Temukan snapshot model yang sudah diunduh. Folder `models--...` adalah root
cache; `HF_MODEL_NAME` harus menunjuk ke folder di bawah `snapshots/`.

```bash
MODEL_ROOT="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
MODEL_REVISION="$(cat "$MODEL_ROOT/refs/main")"
MODEL_PATH="$(readlink -f "$MODEL_ROOT/snapshots/$MODEL_REVISION")"
test -f "$MODEL_PATH/config.json"
echo "$MODEL_PATH"
```

```dotenv
HF_MODEL_NAME=/home/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<commit>
HF_QUANTIZATION=4bit
HF_TRUST_REMOTE_CODE=false
HF_MAX_INPUT_TOKENS=4096
HF_MAX_NEW_TOKENS=768
```

```yaml
generator:
  backend: huggingface
  model_env: HF_MODEL_NAME
  temperature: 0
  max_tokens: 768
  max_tokens_env: HF_MAX_NEW_TOKENS
  quantization: 4bit
  trust_remote_code: false
  max_input_tokens: 4096
  max_input_tokens_env: HF_MAX_INPUT_TOKENS
```

Preset siap pakai berada di `configs/experiments/rankify-multilingual-local.yaml`
dan `configs/experiments/rankify-uniqa-local.yaml`. Snapshot absolut membuat
generator lokal tidak menghubungi Hub, sedangkan dense retriever dan reranker
tetap dapat memakai cache atau mengunduh modelnya sendiri.

Preflight dan benchmark ITS penuh:

```bash
set -a
source .env
set +a
test -f "$HF_MODEL_NAME/config.json"
.venv/bin/python -c 'import os; from transformers import AutoConfig; print(AutoConfig.from_pretrained(os.environ["HF_MODEL_NAME"], local_files_only=True).model_type)'
.venv/bin/python -m rag_benchmark benchmark \
  --config configs/experiments/rankify-multilingual-local.yaml \
  --stage all \
  --output artifacts/benchmarks/its-local-qwen
```

OpenAI-compatible API, termasuk OpenRouter:

```dotenv
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=isi-api-key
LLM_MODEL=openrouter/free
LLM_MODEL_SOURCE=openrouter/free
LLM_MAX_TOKENS=768
```

```yaml
generator:
  backend: openai-compatible
  model_env: LLM_MODEL
  base_url_env: LLM_BASE_URL
  api_key_env: LLM_API_KEY
  temperature: 0
  max_tokens: 768
  max_tokens_env: LLM_MAX_TOKENS
```

API key hanya dibaca dari environment dan tidak dimasukkan ke manifest atau
prediction file. Semua model dapat diganti dari YAML/environment:

```yaml
models:
  dense_multilingual: BAAI/bge-m3
  reranker_multilingual: BAAI/bge-reranker-v2-m3
```

Mengganti model atau konfigurasi menghasilkan fingerprint/index namespace baru.

## GPU dan fallback CPU

RTX 50-series/Blackwell memerlukan wheel PyTorch yang menyertakan `sm_120`.
Setup GPU melakukan smoke test tensor CUDA dan berhenti bila arsitektur GPU tidak
ada pada `torch.cuda.get_arch_list()`:

```bash
bash scripts/setup_rankify_gpu_ubuntu.sh
```

CUDA 13.2 GA dipasangkan dengan driver NVIDIA Linux minimal 595.45.04.
Driver 595.91.07 memenuhi persyaratan ini.

```dotenv
RAG_DEVICE=auto
EMBED_BATCH_SIZE=32
RERANK_BATCH_SIZE=16
```

`auto` memakai CUDA untuk E5, BGE retrieval, dan BGE reranking bila tersedia.
Jika GPU tidak terdeteksi, evaluasi tetap berjalan di CPU dan menulis warning
serta `gpu_fallback` pada diagnostics. Tujuh preset Rankify dijalankan berurutan
agar penggunaan VRAM tetap sesuai untuk GPU sekitar 6 GB.

```bash
uv run --no-sync python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback")'
```

## Dataset

Edit `configs/experiments/rankify-multilingual.yaml` untuk ITS:

```yaml
dataset: its
source: ../../../langchain-chatbot-main/dataset
```

Untuk UniQA:

```yaml
dataset: uniqa
source: ../../../UniQA-main
```

Path relatif dihitung dari lokasi file YAML. Jumlah QA yang dievaluasi dikontrol
oleh konfigurasi berikut:

```yaml
evaluation:
  split: all  # all atau test
```

Kedua preset ITS default ke `all`, sehingga seluruh 340 QA unik dari semua file
`QA Generator*.csv` dan tiga QA CSV top-level dievaluasi. Ubah menjadi `test`
untuk memakai held-out split 20% yang berisi 69 QA. Kedua preset UniQA tetap
default ke `test`, yaitu test split native Inggris dan Italia.

Nilai `all` hanya mengubah QA yang dijadwalkan untuk evaluasi; tidak ada model
training dan index corpus yang sama dapat digunakan kembali. Reference answer
tidak pernah dikirim ke Rankify worker, retriever, prompt, atau generator.

## Menjalankan benchmark

```bash
set -a
source .env
set +a
```

Index saja:

```bash
uv run --no-sync python -m rag_benchmark benchmark \
  --config configs/experiments/rankify-multilingual.yaml \
  --stage index \
  --output artifacts/benchmarks/its-index
```

Evaluasi index yang sudah ada:

```bash
uv run --no-sync python -m rag_benchmark benchmark \
  --config configs/experiments/rankify-multilingual.yaml \
  --stage evaluate \
  --output artifacts/benchmarks/its-run
```

Index lalu evaluasi:

```bash
uv run --no-sync python -m rag_benchmark benchmark \
  --config configs/experiments/rankify-multilingual.yaml \
  --stage all \
  --output artifacts/benchmarks/its-run
```

Di Ubuntu/Linux, tambahkan `-d` atau `--detach` agar proses tetap berjalan
setelah terminal ditutup. Environment `.env` tetap harus di-source sebelum
menjalankan command karena diteruskan ke child process.

```bash
set -a
source .env
set +a

.venv/bin/python -m rag_benchmark benchmark \
  --config configs/experiments/rankify-uniqa-local.yaml \
  --stage all \
  --output artifacts/benchmarks/uniqa-run \
  --detach
```

Command langsung mengembalikan PID serta lokasi log. Mode detached membuat
`benchmark.log` dan `benchmark.pid` di direktori output. Run kedua dengan
direktori output yang sama ditolak selama PID pertama masih aktif.

```bash
tail -f artifacts/benchmarks/uniqa-run/benchmark.log

PID="$(cat artifacts/benchmarks/uniqa-run/benchmark.pid)"
ps -p "$PID" -o pid,ppid,etime,%cpu,%mem,cmd
kill "$PID"
```

Mode ini bertahan setelah terminal ditutup, tetapi tidak setelah mesin reboot.
Output internal worker Rankify masih di-buffer per preset, sehingga log dapat
diam sementara walaupun CPU/GPU masih bekerja.

CLI DiLLeMa lama tetap tersedia:

```bash
uv run --no-sync python -m rag_benchmark index \
  --technique dillema --dataset its \
  --source ../langchain-chatbot-main/dataset \
  --index-root artifacts/indexes

uv run --no-sync python -m rag_benchmark evaluate \
  --technique dillema --dataset its \
  --index-root artifacts/indexes \
  --output artifacts/runs/dillema-its
```

Tidak ada lagi command `train` atau opsi `--checkpoint`.

## Output dan metrik

```text
artifacts/benchmarks/<run-id>/
├── benchmark.log        # hanya mode detached
├── benchmark.pid        # hanya mode detached
├── manifest.json
├── leaderboard.json
├── leaderboard.csv
├── dillema/
│   ├── predictions.jsonl
│   ├── bleu-summary.json
│   └── metrics.json
└── <rankify-preset>/
    ├── predictions.jsonl
    ├── metrics.json
    └── worker-diagnostics.json
```

Evaluator bersama melaporkan BLEU raw/content, Unicode-aware Exact Match dan
token F1, recall@1/5/10, MRR@10, nDCG@10, status/error, serta mean/p50/p95
latency. ITS recall diberi label weak/category-based. Error dan abstention tetap
berada dalam denominator dengan prediction kosong.

Perbandingan dua run hanya diterima bila dataset, split, QA IDs, model/corpus
fingerprint, dan decoding identik:

```bash
uv run --no-sync python -m rag_benchmark compare-techniques \
  --baseline-run artifacts/benchmarks/its-run/dillema/predictions.jsonl \
  --candidate-run artifacts/benchmarks/its-run/rankify_hybrid_bge_rerank_basic/predictions.jsonl \
  --output artifacts/benchmarks/its-run/comparison
```

## Test

```bash
uv run --no-sync pytest
uv run --no-sync python -m rag_benchmark --help
```

Test unit tidak mengunduh model atau memerlukan Rankify. Full benchmark pertama
memerlukan jaringan untuk dependency/model, kemudian dapat memakai cache lokal.

## Referensi

- [Rankify repository](https://github.com/DataScienceUIBK/Rankify)
- [Rankify Pipeline API](https://rankify.readthedocs.io/en/latest/tutorials/pipeline/)
- [Rankify custom indexing](https://rankify.readthedocs.io/en/latest/tutorials/retrieval/custom_datasets/)
- [Rankify transformer reranker](https://rankify.readthedocs.io/en/latest/api/rerankings/transformer_reranker/)
- [SacreBLEU](https://github.com/mjpost/sacreBLEU)
- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
- [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
