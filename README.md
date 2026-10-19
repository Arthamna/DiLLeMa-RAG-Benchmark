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
bash scripts/setup_rankify_ubuntu.sh
cp .env.example .env
```

Kedua environment dibuat oleh `uv` di `.venv` dan `.venv-rankify`, tetap lokal
di repository, dan diabaikan Git. Python mengikuti `.python-version`; worker
dipin pada `rankify==0.1.4`. `uv sync` membuat atau memperbarui lockfile project.

Worker memanggil `TransformerRanker` secara langsung agar registry reranker
Rankify tidak memuat dependency LiT5/RankLLM seperti `fastchat` yang tidak dipakai.
Retriever BM25 dan BGE juga diimpor langsung agar registry retriever tidak
memuat `OnlineRetriever` dan dependency web-search yang tidak digunakan.
Compatibility worker dikunci pada PyArrow 17 dan Chardet 5.x.

Jika `.venv-rankify` sudah dibuat oleh versi setup lama, cukup sinkronkan ulang:

```bash
sudo apt install -y openjdk-21-jdk-headless
export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")"
java -version
javac -version

export UV_PROJECT_ENVIRONMENT="$PWD/.venv-rankify"
uv sync --project rankify_worker --python 3.12 --locked

printf '%s\n' '{"schema_version":1,"operation":"capabilities"}' \
  | .venv-rankify/bin/python -m rag_benchmark_rankify.worker
```

Worker mendeteksi JDK dari `javac` dan mengatur `JAVA_HOME` untuk subprocess-nya.
Probe yang berhasil berakhir dengan `"status": "ok"` dan menampilkan versi
Python, NumPy, PyArrow, Torch, lokasi Java, serta status CUDA.

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

Path relatif dihitung dari lokasi file YAML. ITS memakai held-out split 20%;
UniQA memakai test split native Inggris dan Italia. Reference test tidak pernah
dikirim ke Rankify worker, retriever, prompt, atau generator.

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
