# DiLLeMa–Rankify–Qwen3 RAG Benchmark

Benchmark terkontrol untuk membandingkan full conventional RAG DiLLeMa dengan
tujuh pipeline Rankify dan dua varian Qwen3 pada corpus ITS dan UniQA. Indexing membangun retrieval
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

## Struktur implementasi

Implementasi setiap teknik berada di `src/rag_benchmark/techniques/`:

| Lokasi | Tanggung jawab |
|---|---|
| `techniques/dillema/` | E5, sparse Qdrant, hybrid retrieval, reranking, filtering, parent packing, prompt, dan runner DiLLeMa |
| `techniques/rankify/` | Preset, client subprocess, ekspor corpus, dan pengolahan hasil Rankify |
| `techniques/rankify/worker/` | Proyek Python terpisah berisi dependency dan implementasi worker Rankify |
| `techniques/qwen3/` | Qwen3 embedding, dense cosine index, yes/no reranker, prompt, dan staged runner |
| `shared/` | Records, dataset, chunking semua teknik, evaluasi, metrik, dan runtime LLM |
| `cli.py` dan `benchmark.py` | Entrypoint command, konfigurasi eksperimen, penjadwalan teknik, dan leaderboard |

Di dalam DiLLeMa, `pipeline.py` mengatur retrieval/filtering, `augmentation.py`
menyusun parent context, dan `generation.py` membentuk prompt. Runtime pada
`shared/api_llm.py` dan `shared/local_llm.py` menerima prompt melalui
`generate_prompt(prompt)`; adapter `DiLLeMaGenerator` menyediakan
`generate(question, context)`.

Worker tetap dipasang sebagai distribution `dillema-rankify-worker`, dengan
package `rag_benchmark_rankify` di environment `.venv-rankify`. Proyek worker
dikecualikan dari package discovery utama dan dibangun secara terpisah.
Proses utama berkomunikasi dengan worker melalui JSONL tanpa mengimpor library
Rankify. Import shared tidak bergantung pada implementasi teknik, dan kedua
teknik tidak saling mengimpor.

Contoh import baru:

```python
from rag_benchmark.shared.chunking import DocumentChunker
from rag_benchmark.shared.config import ChunkingConfig
from rag_benchmark.shared.metrics import summarize
from rag_benchmark.shared.records import DocumentRecord
from rag_benchmark.techniques.dillema.config import PipelineConfig
from rag_benchmark.techniques.dillema.pipeline import RAGPipeline
from rag_benchmark.techniques.rankify.client import RankifyWorkerClient
```

Import lama seperti `rag_benchmark.pipeline`, `rag_benchmark.store`, dan
`rag_benchmark.metrics` sudah dihapus. Format YAML, nama command dan preset,
fingerprint, serta lokasi indeks/output tetap. Di YAML eksperimen, bagian
`models` dan `retrieval` mengatur Rankify; konfigurasi DiLLeMa berada pada
`techniques/dillema/config.py` dan environment yang sudah didokumentasikan.

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
# Opsional, hanya bila memakai server vLLM lokal:
bash scripts/setup_vllm_gpu_ubuntu.sh
cp .env.example .env
```

Environment utama dan GPU Rankify dibuat oleh `uv` di `.venv` dan
`.venv-rankify-gpu`; server vLLM opsional dibuat terpisah di
`.venv-vllm-gpu`. Ketiganya tetap lokal di repository dan diabaikan Git. Python
mengikuti `.python-version`; worker dipin pada `rankify==0.1.4`. Environment
Rankify GPU memakai `torch==2.13.0+cu132` dan `bitsandbytes==0.50.2`, termasuk
kernel Blackwell `sm_120` untuk RTX 50-series. Konfigurasi eksperimen bawaan
menunjuk ke `.venv-rankify-gpu/bin/python`.

Worker memanggil `TransformerRanker` secara langsung agar registry reranker
Rankify tidak memuat dependency LiT5/RankLLM seperti `fastchat` yang tidak dipakai.
Retriever BM25 dan BGE juga diimpor langsung agar registry retriever tidak
memuat `OnlineRetriever` dan dependency web-search yang tidak digunakan.
Compatibility worker dikunci pada PyArrow 17 dan Chardet 5.x.
Environment GPU tidak memasang extra reranking Rankify karena extra tersebut
menarik vLLM 0.7.0 dan PyTorch 2.5.1 yang tidak dipakai oleh direct
`TransformerRanker` serta tidak mendukung `sm_120`.
Rankify 0.1.4 tetap mengimpor kelas vLLM secara eager ketika generator dimuat.
Worker menyediakan import shim terbatas untuk kasus tersebut; shim akan gagal
dengan pesan eksplisit bila backend vLLM in-process benar-benar dipilih. Engine
vLLM nyata hanya dijalankan dari `.venv-vllm-gpu` melalui API lokal.

Untuk GPU lama atau fallback kompatibilitas, `.venv-rankify` tetap tersedia:

```bash
sudo apt install -y openjdk-21-jdk-headless
export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")"
java -version
javac -version

export UV_PROJECT_ENVIRONMENT="$PWD/.venv-rankify"
uv sync --project src/rag_benchmark/techniques/rankify/worker \
  --python 3.12 --extra legacy --locked --reinstall-package dillema-rankify-worker

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

### vLLM lokal melalui API OpenAI-compatible

Setup vLLM hanya perlu dilakukan satu kali dan tidak mengubah dependency
Rankify:

```bash
bash scripts/setup_vllm_gpu_ubuntu.sh
```

Isi `.env` dengan model dan token lokal. `LLM_API_KEY` di sini adalah token
pilihan sendiri untuk melindungi endpoint lokal, bukan API key provider.

```dotenv
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
LLM_API_KEY=local-vllm-token
LLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_GPU_MEMORY_UTILIZATION=0.65
```

Muat `.env` pada terminal server:

```bash
set -a
source .env
set +a
bash scripts/run_vllm_gpu_ubuntu.sh
```

Pada terminal lain, periksa server dan jalankan preset OpenAI-compatible yang
sudah ada:

```bash
curl -fsS \
  -H "Authorization: Bearer $LLM_API_KEY" \
  "$LLM_BASE_URL/models"

set -a
source .env
set +a
.venv/bin/python -m rag_benchmark benchmark \
  --config configs/experiments/rankify-multilingual.yaml \
  --stage all \
  --output artifacts/benchmarks/its-vllm
```

Jangan jalankan server vLLM dan generator Hugging Face lokal bersamaan pada
GPU 16 GB. Nilai `VLLM_GPU_MEMORY_UTILIZATION` dapat disesuaikan bila model dan
retriever/reranker memerlukan pembagian VRAM berbeda. Wheel vLLM 0.32.0 memakai
runtime CUDA 13.0 dan dijalankan oleh driver CUDA 13.2 melalui kompatibilitas
driver NVIDIA.

## GPU dan fallback CPU

RTX 50-series/Blackwell memerlukan wheel PyTorch yang menyertakan `sm_120`.
Environment utama dan Rankify GPU dikunci ke PyTorch 2.13.0 CUDA 13.2. Kedua
setup melakukan smoke test tensor CUDA dan berhenti bila arsitektur GPU tidak
ada pada `torch.cuda.get_arch_list()`:

```bash
bash scripts/setup_ubuntu.sh
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

Pipeline menyinkronkan CUDA sesudah E5 embedding, BGE reranking, dan generation.
Jika kernel gagal, log menyebut stage tersebut dan evaluasi berhenti sebelum
sticky error muncul lagi di `empty_cache()`. Untuk diagnosis satu kali dengan
eksekusi kernel sepenuhnya synchronous:

```bash
CUDA_LAUNCH_BLOCKING=1 .venv/bin/python -m rag_benchmark benchmark \
  --config configs/experiments/rankify-uniqa-local.yaml \
  --stage evaluate \
  --output artifacts/benchmarks/uniqa-debug
```

Setelah `CUDA_ERROR_LAUNCH_FAILED`/error 719, hentikan proses yang gagal lalu
jalankan proses baru; CUDA context pada proses lama tidak dapat dipulihkan.

Jika error terjadi pada `qwen3_embedding generation`, operasi yang gagal adalah
generator bersama, bukan model embedding/reranker Qwen3 yang sudah dilepas.
Untuk generator **Qwen2.5-0.5B-Instruct pada RTX 5080**, coba BF16 tanpa
quantization dan eager attention dalam proses baru:

```bash
set -a
source .env
set +a
export HF_QUANTIZATION=none
export HF_TORCH_DTYPE=bfloat16
export HF_ATTENTION_IMPLEMENTATION=eager
test -f "$HF_MODEL_NAME/config.json"
bash scripts/setup_rankify_gpu_ubuntu.sh
```

`HF_MODEL_NAME` tetap harus menunjuk snapshot berisi `config.json`, bukan parent
`models--Qwen--Qwen2.5-0.5B-Instruct`. YAML lokal memakai
`quantization_env: HF_QUANTIZATION`, sehingga export di atas mengalahkan default
`quantization: 4bit`. Dtype dan attention dapat diatur juga lewat
`generator.torch_dtype` dan `generator.attention_implementation`; nilai YAML
eksplisit mengalahkan environment. Semua setting ini masuk fingerprint generator
dan diteruskan juga ke worker Rankify. `qwen3.attention_implementation` hanya
mengontrol model retrieval Qwen3.

BF16 tanpa quantization dan eager attention merupakan langkah isolasi kernel,
bukan diagnosis pasti error 719. Eager attention dapat lebih lambat dan memakai
VRAM lebih besar. Untuk memeriksa langsung QA berikutnya setelah progres terakhir
333/340 tanpa mengulang seluruh matriks, gunakan index yang sudah dibangun:

```bash
CUDA_LAUNCH_BLOCKING=1 .venv/bin/python - <<'PY'
import logging
from dataclasses import replace
from pathlib import Path
from rag_benchmark.benchmark import ExperimentSpec
from rag_benchmark.shared.datasets import load_dataset
from rag_benchmark.techniques.qwen3.runner import run_evaluate

logging.basicConfig(level=logging.INFO)
spec = ExperimentSpec.load(Path("configs/experiments/rankify-multilingual-local.yaml"))
bundle = load_dataset(spec.dataset, spec.source)
questions = bundle.questions_for_split(spec.evaluation_split)
last_completed = "its-qa-44097ea6bdf7deb65b8dceaf"
position = next(i for i, q in enumerate(questions) if q.id == last_completed)
question = questions[position + 1]
print("Testing next QA:", question.id, flush=True)
run_evaluate(spec, replace(bundle, questions=[question]),
             Path("artifacts/benchmarks/its-qwen3-cuda-probe"), variants=("qwen3_embedding",))
PY
```

Jika probe berhasil, jalankan ulang benchmark ke direktori output baru agar
hasil 4-bit sebelumnya tidak tertimpa atau tercampur dengan BF16. Jangan resume
prediksi lama dengan fingerprint generator yang berbeda. Jika gagal lagi,
traceback kini menyertakan QA ID, stage, input token count, dan setting generator.
Periksa juga `journalctl -k -b | grep -Ei 'NVRM|Xid'` untuk diagnosis driver/GPU.

Referensi: [Transformers attention/model loading](https://huggingface.co/docs/transformers/v4.57.1/main_classes/model),
[konfigurasi resmi Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json),
dan [NVIDIA CUDA error 719](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html).

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

## Qwen3 Embedding dan Reranker

Keempat YAML eksperimen sekarang mengaktifkan dua varian Qwen3, sehingga satu
command `benchmark` menghasilkan sepuluh pipeline. Model generator tetap dipilih
oleh blok `generator`; Qwen3 retrieval tidak mengganti generator Qwen2.5 Anda.
ITS tetap default `evaluation.split: all`, UniQA tetap native `test`.

```yaml
retrieval:
  candidates: 100
  top_k: 10
  rrf_k: 60

qwen3:
  enabled: true
  embedding_model: Qwen/Qwen3-Embedding-0.6B
  reranker_model: Qwen/Qwen3-Reranker-4B
  instruction: Given a question, retrieve relevant passages that answer the question.
  embedding_dimension: 1024
  embedding_max_length: 2048
  reranker_max_length: 2048
  embedding_batch_size: 8
  reranker_batch_size: 1
  torch_dtype: bfloat16
  reranker_quantization: none
  attention_implementation: sdpa
```

`qwen3_embedding` menggunakan top-10 cosine dari top-100 dense candidates.
`qwen3_embedding_rerank` meranking ulang **pool top-100 yang sama**, lalu mengambil
top-10 dengan probabilitas relevansi `yes/no`. Kedua varian memakai prompt dan
context packing yang sama. `retrieval.candidates` juga mengontrol budget kandidat
DiLLeMa dan Rankify pada benchmark; algoritme hybrid/filtering/parent packing
DiLLeMa tetap berlaku. Ini perbandingan full pipeline, bukan isolasi seluruh
perbedaan algoritme antarteknik. Ablation dua varian Qwen3 mengisolasi reranking.

Untuk RTX 5080 16 GB, runner menjalankan retrieval seluruh QA, melepas embedding,
menjalankan reranking, melepas reranker, lalu baru memuat generator dari runtime
bersama. Cache kandidat sementara tidak berisi reference answer dan dihapus
setelah evaluasi. Prediction ditulis per QA dan log menunjukkan progres setiap
tahap. Jika reranker 4B BF16 mengalami OOM, ubah `reranker_quantization: 4bit`;
ukuran model, revision, precision, dan quantization dicatat pada artifact.
CPU fallback memakai float32 tanpa quantization GPU dan dicatat di diagnostics.
Server generator eksternal seperti vLLM tetap memakai VRAM selama server hidup;
staged runner hanya mengelola model dalam proses benchmark.

Model ID di YAML otomatis memakai cache/download Hugging Face. Untuk snapshot
lokal, ganti `embedding_model`/`reranker_model` dengan absolute path berisi
`config.json`; loader memakai `local_files_only=True`. Opsional, pin
`embedding_revision` dan `reranker_revision` ke commit Hugging Face. Defaultnya
`main`; jika resolved embedding revision berubah sejak indexing, evaluator
meminta rebuild. Untuk error token publik kedaluwarsa, muat environment dengan
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` sebagaimana contoh `.env.example`.

Setup memakai `.venv` utama (Transformers >=4.51) dan worker Rankify terisolasi:

```bash
bash scripts/setup_ubuntu.sh
bash scripts/setup_rankify_gpu_ubuntu.sh
set -a
source .env
set +a

.venv/bin/python -m rag_benchmark benchmark \
  --config configs/experiments/rankify-uniqa-local.yaml \
  --stage all \
  --output artifacts/benchmarks/uniqa-qwen3
```

Gunakan preset `rankify-multilingual-local.yaml` untuk ITS atau preset tanpa
`-local` untuk API generator. `--detach` tetap didukung. Perubahan budget kandidat
40 menjadi 100 mengubah namespace index lama, jadi jalankan `--stage all` sekali;
setelah itu `--stage evaluate` dapat memakai index yang sama. Index Qwen3 ada di
`artifacts/indexes/qwen3/<dataset>/<hash>/`; mengganti generator/reranker tidak
membangun ulang embedding index. `qwen3.enabled: false` mengembalikan matriks
delapan pipeline; YAML lama tanpa blok ini juga tetap diterima.

Output tambahan:

```text
qwen3_embedding/{predictions.jsonl,metrics.json,manifest.json,diagnostics.json}
qwen3_embedding_rerank/{predictions.jsonl,metrics.json,manifest.json,diagnostics.json}
qwen3-ablation.json
```

Setiap folder juga berisi `bleu-summary.json` dan `missing-context.jsonl`.
Evaluator bersama menghitung BLEU raw/content, EM, token F1, Recall@1/5/10,
MRR@10, nDCG@10, latency, error, dan fallback. Qwen3 menambahkan candidate
Recall@100 (hit setidaknya satu source relevan, sesuai definisi recall evaluator
lama); label ITS tetap `weak/category-based`. Laporan ablation memverifikasi hash
pool kandidat dan memberikan delta per bahasa. Latency Qwen3 merupakan jumlah
tahap per QA, tidak termasuk waktu load model dan antrean batch; biaya retrieval
bersama diatribusikan pada kedua varian. Latency run total dapat lebih lama.

Teknik Qwen3 juga tersedia lewat `index` dan `evaluate` dengan
`--technique qwen3_embedding|qwen3_embedding_rerank --config <experiment.yaml>`.
Untuk command ini `--dataset` wajib cocok dengan YAML; `index` tetap membutuhkan
`--source`. `--index-root` dapat meng-override YAML. Evaluasi satu varian tetap
mengikuti `evaluation.split`. `compare-techniques` menerima prediction Qwen3
dengan syarat corpus, QA, generator, split, dan decoding sama.

Referensi: [repo Qwen3](https://github.com/QwenLM/Qwen3-Embedding),
[Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), dan
[Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B). Hasil retrieval MTEB
resmi bukan jaminan kenaikan BLEU ITS/UniQA; ukur delta pada corpus Anda sendiri.

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
