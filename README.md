# DiLLeMa Multi-Technique RAG Benchmark

Standalone benchmark for two RAG techniques over the ITS and UniQA corpora:

- `dillema`: structured chunking, multilingual E5 dense plus sparse/IDF Qdrant RRF,
  multilingual cross-encoder reranking, relative filtering, and parent packing.
- `rrag`: hybrid candidate retrieval followed by the three official R²AG retrieval
  features, a trainable R²-Former, retrieval-aware prompting, and retrieval-embedding
  injection into a frozen local causal LLM.

By default both techniques use the same Hugging Face base LLM and deterministic
decoding. DiLLeMa can optionally generate through an OpenAI-compatible API; R²AG
still requires the local model because it injects retrieval embeddings into the LLM.
The project excludes frontend, web API, authentication, conversation memory,
adaptive routing, query augmentation, scope gates, and knowledge graphs.

## Architecture

```text
src/rag_benchmark/
├── shared/                 dataset, records, retrieval facade, local LLM, evaluation
└── techniques/
    ├── dillema/            conventional full DiLLeMa RAG
    └── rrag/               features, R²-Former, trainer, and RRAG pipeline
```

The local base-model loader, tokenizer, quantization, prompt decoding, and generation
live in `shared/local_llm.py`; the optional API runtime lives in `shared/api_llm.py`.
R²-Former is technique-specific and remains under `techniques/rrag/`.

## Setup di Ubuntu Linux

Python 3.10 atau lebih baru diperlukan. Instal prasyarat sistem dan pastikan driver
NVIDIA aktif bila memakai GPU:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential
nvidia-smi
```

Jalankan setup dari root repository. Skrip hanya membuat `.venv` bila belum ada dan
memasang seluruh dependency secara lokal di dalam repository:

```bash
chmod +x scripts/setup_ubuntu.sh
./scripts/setup_ubuntu.sh
source .venv/bin/activate
```

Gunakan `INSTALL_QUANTIZED=0 ./scripts/setup_ubuntu.sh` jika tidak ingin memasang
`bitsandbytes`. Setup manual yang ekuivalen:

```bash
test -d .venv || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[api,dev,rrag,quantized]"
```

Salin konfigurasi, pilih generator lokal atau API, lalu ekspor isinya. CLI tidak
memuat `.env` secara otomatis.

```bash
cp -n .env.example .env
nano .env
set -a
source .env
set +a
```

Untuk generator Hugging Face lokal, konfigurasi minimalnya:

```dotenv
HF_MODEL_NAME=your-huggingface-causal-lm
HF_QUANTIZATION=4bit
RAG_DEVICE=auto
```

Untuk menggunakan OpenRouter atau endpoint OpenAI-compatible lain pada evaluasi
DiLLeMa, isi konfigurasi berikut. `LLM_API_KEY` yang tidak kosong otomatis memilih
generator API dan `HF_MODEL_NAME` tidak diperlukan untuk evaluasi DiLLeMa tersebut.

```dotenv
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your-api-key
LLM_MODEL=openrouter/free
LLM_MODEL_SOURCE=openrouter/free
LLM_MAX_TOKENS=768
```

R²AG tetap memerlukan `HF_MODEL_NAME` untuk training dan evaluasi karena retrieval
embedding disisipkan langsung ke hidden state model lokal. Kosongkan `LLM_API_KEY`
untuk kembali memakai Hugging Face lokal pada DiLLeMa. Jangan commit file `.env`.

There is deliberately no default LLM. Optional settings are:

- `HF_QUANTIZATION=4bit|8bit|none` (default `4bit`)
- `HF_TRUST_REMOTE_CODE=false`
- `HF_MAX_INPUT_TOKENS=4096`
- `HF_MAX_NEW_TOKENS=768`
- `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_MODEL_SOURCE`, dan
  `LLM_MAX_TOKENS` untuk generator API opsional DiLLeMa
- `RAG_DEVICE=auto|cuda|cuda:0|cpu` (default `auto`)
- `EMBED_BATCH_SIZE=32`
- `RERANK_BATCH_SIZE=16`
- `LOG_LEVEL=INFO`
- `EMBED_MODEL_NAME=intfloat/multilingual-e5-base`
- `RERANK_MODEL_NAME=BAAI/bge-reranker-v2-m3`
- `RRAG_TOP_K=10`, `RRAG_EPOCHS=2`, `RRAG_LEARNING_RATE=2e-4`
- `QDRANT_URL` and `QDRANT_API_KEY` for remote Qdrant

Dengan `RAG_DEVICE=auto`, E5 dan BGE reranker memakai CUDA secara eksplisit jika
tersedia dan log akan menyebut nama GPU. Jika CUDA tidak terdeteksi, pipeline tetap
berjalan di CPU dengan warning; quantization LLM 4/8-bit juga dilewati agar fallback
CPU tidak langsung gagal. Untuk GPU 6 GB, gunakan instruction model kecil,
`HF_QUANTIZATION=4bit`, dan turunkan `RERANK_BATCH_SIZE` bila kehabisan VRAM.

```bash
which python
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback")'
```

## Dataset protocol

ITS indexing reads only the 18 category `Dataset IF-Tegration - *.csv` files and
produces 53 deduplicated documents. Its 340 unique QA records are split
deterministically and by category into 271 train and 69 held-out test records.
Folder membership supplies weak document-relevance labels; answer text is never used
to choose relevant documents.

UniQA indexes 522 English and 521 Italian TXT files. It uses each language's native
5,298 train and 1,573 test records. The strict test denominator retains the 42 English
and 14 Italian questions whose source context is absent.

Training answers may be used only by the R²-Former training loss. Test answers remain
evaluation-only and never enter documents, indexes, retrieval queries, prompts, or
generation context.

## Commands

Indexes are isolated under `<index-root>/<technique>/<dataset>/`.

```bash
source .venv/bin/activate

for technique in dillema rrag; do
  python -m rag_benchmark index \
    --technique "$technique" --dataset its \
    --source ../langchain-chatbot-main/dataset \
    --index-root artifacts/indexes

  python -m rag_benchmark index \
    --technique "$technique" --dataset uniqa \
    --source ../UniQA-main \
    --index-root artifacts/indexes
done
```

Indexing only builds retrieval data. It is not model training. R²AG additionally
trains its small auxiliary R²-Former and projection while freezing the retriever and
base LLM:

```bash
python -m rag_benchmark train \
  --technique rrag --dataset uniqa \
  --index-root artifacts/indexes \
  --output artifacts/models/rrag/uniqa

python -m rag_benchmark train \
  --technique rrag --dataset its \
  --index-root artifacts/indexes \
  --output artifacts/models/rrag/its
```

The command saves one auxiliary checkpoint per language. DiLLeMa intentionally rejects
the `train` command.

```bash
python -m rag_benchmark evaluate \
  --technique dillema --dataset uniqa \
  --index-root artifacts/indexes \
  --output artifacts/runs/dillema-uniqa

python -m rag_benchmark evaluate \
  --technique rrag --dataset uniqa \
  --index-root artifacts/indexes \
  --checkpoint artifacts/models/rrag/uniqa \
  --output artifacts/runs/rrag-uniqa
```

Untuk ITS, ubah `--dataset` menjadi `its`, gunakan checkpoint
`artifacts/models/rrag/its`, dan pilih folder output ITS yang berbeda.

Compare runs only when dataset, held-out QA IDs, split, base-model fingerprint, and
decoding configuration match:

```bash
python -m rag_benchmark compare-techniques \
  --baseline-run artifacts/runs/dillema-uniqa/predictions.jsonl \
  --candidate-run artifacts/runs/rrag-uniqa/predictions.jsonl \
  --output artifacts/runs/comparison
```

The existing corpus comparison remains available as `compare --its-run ...
--uniqa-run ...`. Its report keeps Indonesian, English, and Italian separate and must
not be interpreted as proving that one corpus is intrinsically better.

## Evaluation artifacts

Each evaluation writes:

```text
manifest.json
predictions.jsonl
bleu-summary.json
missing-context.jsonl
config.json
```

Every held-out QA produces exactly one prediction. Error, abstention, no-evidence, and
missing-context records stay in the BLEU denominator with an empty prediction. Reports
contain mixed-case SacreBLEU corpus BLEU-4 with tokenizer `13a`, `BLEU_raw`,
`BLEU_content`, source recall@10, average latency, and language-specific UniQA strict
and corpus-valid results. ITS recall is explicitly marked as weak category relevance.

Prediction diagnostics contain DiLLeMa filtering/parent counts or RRAG retrieval
features and R²-Former relevance probabilities. API keys and model weights are never
written to run metadata.

## Tests

```bash
source .venv/bin/activate
python -m pytest
```

Tests cover adapter schemas and canonical counts, deterministic split isolation,
multilingual listing detection, dense+sparse RRF, DiLLeMa regression behavior, all
three R²AG features, R²-Former shapes/checkpoint round trips when the optional Torch
extra is installed, BLEU edge cases, and in-memory end-to-end smoke paths.

## References

- [R²AG paper, Findings of EMNLP 2024](https://doi.org/10.18653/v1/2024.findings-emnlp.678)
- [Official yefd/RRAG repository](https://github.com/yefd/RRAG)
- [Official R²AG model implementation](https://github.com/yefd/RRAG/blob/main/RRAG/models/modeling_rrag.py)
- [Official retrieval feature implementation](https://github.com/yefd/RRAG/blob/main/retrieval/retrieval_utils.py)
- [Qdrant hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/)
- [SacreBLEU](https://github.com/mjpost/sacreBLEU)
- [Python `venv` documentation](https://docs.python.org/3/library/venv.html)
- [PyTorch CUDA availability](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html)
- [Sentence Transformers `CrossEncoder`](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html)
- [OpenAI Python client](https://github.com/openai/openai-python)
- [OpenRouter API quickstart](https://openrouter.ai/docs/quickstart)

See `THIRD_PARTY_NOTICES.md` for RRAG attribution.

