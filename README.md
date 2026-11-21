# DiLLeMa RAG Benchmark

Branch `feat/old-90` juga menyediakan preset **DiLLeMa, BGE-M3, Self-RAG,
dan RankZephyr**: 10 varian, split `test`, 40 kandidat retrieval, dan output
maksimum 128 token. Setup, smoke test GPU, full run ITS/UniQA, serta command
per keluarga tersedia di [Runbook empat teknik](RUN.md#dillema-bge-m3-self-rag-dan-rankzephyr-pada-featold-90).
Preset ini tidak membutuhkan Java atau layanan Docker RETRO/PipeRAG.

Matriks RETRO/PipeRAG yang sudah ada tetap tersedia pada ITS dan UniQA:

- `dillema`: pipeline RAG dokumen yang sudah ada.
- `retro`: checkpoint RETRO dengan retrieval fresh/sinkron setiap 64 token.
- `piperag`: checkpoint RETRO yang sama dengan retrieval stale/asinkron dan
  pemilihan `nprobe` dinamis berdasarkan profil hardware.

RETRO dan PipeRAG adalah pasangan terkontrol. Perbandingan DiLLeMa dengan
RETRO/PipeRAG tetap berguna, tetapi ditandai sebagai perbandingan lintas model.

## Arsitektur

```text
benchmark host
  ├─ DiLLeMa (lokal/API)
  ├─ retrieval:50051 (CPU, MiniLM + FAISS IndexIVFFlat)
  └─ inference:50052 (satu GPU, RETRO encoder/decoder ONNX)
```

Runtime tidak mengimpor `reference/`. Model inference minimum dari PipeRAG
commit `b6eb158` di-vendor di `src/rag_benchmark/_vendor/piperag/` dengan
Apache-2.0; lineage RETRO dikunci pada commit `6945cdd`.

## Prasyarat

- Linux, Docker Compose, NVIDIA driver 580 atau lebih baru, dan NVIDIA Container Toolkit.
- Satu GPU NVIDIA untuk service inference.
- Python 3.11–3.12 dan `uv` untuk CLI benchmark host.
- Checkpoint RETRO `model.ckpt` dan konfigurasi `retro.json`, disediakan manual.
- Kredensial/model DiLLeMa melalui backend Hugging Face atau API kompatibel OpenAI.

Checkpoint, model weights, ONNX, index, dan hasil benchmark tidak masuk Git.
SHA-256 checkpoint/ONNX direkam di manifest.

Stack GPU mengikuti patch RTX 50-series dari `feat/rankify-50`: PyTorch
`2.13.0+cu132`, CUDA 13.2/cuDNN 9, dan ONNX Runtime GPU 1.27. RTX 5080
terdeteksi sebagai `sm_120`; setup dan health check menolak wheel yang tidak
memuat arsitektur tersebut. Image retrieval tetap memakai Torch CPU-only.

## Setup

```bash
cp .env.example .env
mkdir -p models artifacts/onnx artifacts/indexes artifacts/huggingface
cp /path/to/model.ckpt models/model.ckpt
cp /path/to/retro.json models/retro.json
bash scripts/setup_ubuntu.sh
docker compose up --build -d
```

Isi `LLM_BASE_URL`, `LLM_API_KEY`, dan `LLM_MODEL` di `.env`, lalu:

```bash
set -a; source .env; set +a
uv run --no-sync python -m rag_benchmark doctor \
  --config configs/experiments/its.yaml
```

`doctor` memeriksa versi protokol, mount checkpoint, CUDA, provider ONNX,
fingerprint artifact, dan kesehatan kedua service. Export ONNX hanya diulang jika
hash checkpoint, konfigurasi, kode runtime, atau versi Torch/ONNX berubah. Sebelum artifact siap,
token argmax dan logit PyTorch-versus-ONNX harus lolos toleransi numerik.

## Menjalankan benchmark

```bash
uv run --no-sync python -m rag_benchmark benchmark \
  --config configs/experiments/its.yaml \
  --stage all \
  --output artifacts/benchmarks/its

uv run --no-sync python -m rag_benchmark benchmark \
  --config configs/experiments/uniqa.yaml \
  --stage all \
  --output artifacts/benchmarks/uniqa
```

`--techniques` hanya menerima `dillema,retro,piperag`. Stage `all` membangun dua
index (QA dan paper hold-out), mengalibrasi profil, menjalankan QA, perplexity,
dan sweep latency. Stage `index` dan `evaluate` dapat dipakai terpisah.

Preset resmi menetapkan greedy decoding dan maksimum 128 token jawaban untuk
semua teknik. Prompt RETRO/PipeRAG berbentuk `Question: …\nAnswer:`, ditrunkasi
dan di-left-pad tepat 64 token.

## Output

Setiap teknik menghasilkan `predictions.jsonl`, `metrics.json`, `config.json`,
dan `manifest.json`. Root run berisi leaderboard JSON/CSV. Paper evaluation
menulis `paper/pareto.json`, `paper/pareto.csv`, `paper/pareto.png`, dan profil.

Diagnostics per QA mencakup token prompt/generasi, token per detik, jumlah
retrieval, interval/staleness, fixed/dynamic `nprobe`, serta latency embedding,
search, wait, encoder, decoder, dan end-to-end.

Hold-out paper adalah 10% dokumen per bahasa berdasarkan SHA-256 seed 42,
minimal satu dokumen, dan tidak pernah masuk index paper. Perplexity memakai
maksimum 32 window × 1.024 token per bahasa. DiLLeMa tidak diberi perplexity
karena backend generiknya tidak menjamin logit yang sebanding.

## Verifikasi

```bash
uv run --no-sync pytest
docker compose ps
docker compose logs retrieval inference
```

Host pengembangan tanpa GPU tetap dapat menjalankan unit/contract test. Full
acceptance, export/parity ONNX, dan angka Pareto harus dijalankan pada mesin GPU;
angka latency adalah hasil hardware lokal, bukan klaim reproduksi absolut
konfigurasi dua-host dari paper PipeRAG.

Lihat [RUN.md](RUN.md) untuk checklist operasional dan
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) untuk atribusi.
