# DiLLeMa RAG Benchmark

Standalone extraction of DiLLeMa's controlled conventional RAG pipeline for comparing the ITS corpus with UniQA English and Italian:

```text
load corpus -> clean/parse -> structured chunking
-> multilingual E5 dense + hashed sparse/IDF indexing
-> Qdrant reciprocal-rank fusion -> bge-reranker-v2-m3
-> absolute and relative evidence filtering -> parent packing
-> OpenAI-compatible generation -> SacreBLEU
```

This project deliberately excludes the DiLLeMa frontend, FastAPI, PostgreSQL, authentication, conversation memory, adaptive/agentic routing, query augmentation, scope gates, and knowledge graphs. **Indexing builds the RAG index; it is not model fine-tuning.** Reference answers are used only after generation for evaluation and never enter indexing, retrieval, context, prompts, or the generator.

## Controlled defaults

Both datasets use exactly the same retrieval settings: 800-character chunks, 120-character overlap, 40 hybrid candidates, reranker floor 0.05, relative floor 0.5, and at most four parent documents. Indonesian, English, and Italian listing questions widen this to 16 reranked passages, relative floor 0.3, and eight parents. Sparse terms use Unicode surface forms with no Sastrawi stemming.

Collections are isolated as `its`, `uniqa_en`, and `uniqa_it`; parallel UniQA documents therefore cannot cross languages.

## Setup

Python 3.10 or newer is required.

```powershell
cd DiLLeMa-RAG-Benchmark
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Export these variables in the shell that runs evaluation (the CLI does not automatically load `.env`):

```text
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=local-key
LLM_MODEL=your-openai-compatible-model
```

Optional variables:

- `EMBED_MODEL_NAME` (default `intfloat/multilingual-e5-base`)
- `RERANK_MODEL_NAME` (default `BAAI/bge-reranker-v2-m3`)
- `QDRANT_URL` and `QDRANT_API_KEY` for a remote server. Without `QDRANT_URL`, indexes persist locally under `<index-root>/qdrant`.

The first real index/evaluation downloads the embedding and reranking models unless they are already cached.

## Index

Run from this project directory. Indexing recreates only the selected benchmark collections.

```powershell
python -m rag_benchmark index `
  --dataset its `
  --source ..\langchain-chatbot-main\dataset `
  --index-root artifacts\indexes

python -m rag_benchmark index `
  --dataset uniqa `
  --source ..\UniQA-main `
  --index-root artifacts\indexes
```

The canonical adapters expect and verify through tests:

| Dataset | Indexed documents | Evaluation QA | Missing-source QA |
|---|---:|---:|---:|
| ITS | 53 deduplicated from 145 rows | 340 unique from 383 rows | 0 |
| UniQA English | 522 TXT | 1,573 test | 42 |
| UniQA Italian | 521 TXT | 1,573 test | 14 |

ITS reads only the 18 category `Dataset IF-Tegration - *.csv` files. It ignores top-level corpus duplicates, `per files dataset`, `txt files`, and evaluation-result CSVs. The 18 category QA Generator files plus three approved top-level QA files are merged; duplicate questions preserve distinct answers as BLEU references.

## Evaluate

The source path is read from the index manifest. `--source` is available only as an explicit relocation override.

```powershell
python -m rag_benchmark evaluate `
  --dataset its `
  --index-root artifacts\indexes `
  --output artifacts\runs\its-run

python -m rag_benchmark evaluate `
  --dataset uniqa `
  --index-root artifacts\indexes `
  --output artifacts\runs\uniqa-run
```

Generation is deterministic at `temperature=0`, `max_tokens=768`, and requests answers without citation markers. Every scheduled QA produces exactly one prediction row. An inference error, abstention, no-evidence result, or missing UniQA source remains in the strict denominator with an empty prediction and therefore contributes zero BLEU credit.

SacreBLEU reports mixed-case corpus BLEU-4 with tokenizer `13a` and stores its reproducibility signature. Each report includes:

- `BLEU_raw`, against original references;
- `BLEU_content`, after deterministic removal of URLs and fixed “for more information” boilerplate in English, Italian, and Indonesian;
- ITS as one Indonesian result;
- UniQA `strict` over all 3,146 test QA;
- UniQA `corpus_valid` over 1,531 English and 1,559 Italian QA;
- English, Italian, and unweighted language macro-average separately.

Do not interpret scores across different languages as proof that one corpus is intrinsically better.

## Compare

```powershell
python -m rag_benchmark compare `
  --its-run artifacts\runs\its-run\predictions.jsonl `
  --uniqa-run artifacts\runs\uniqa-run\predictions.jsonl `
  --output artifacts\runs\comparison
```

This writes `comparison.json` with side-by-side language-specific results and the cross-language interpretation warning.

## Run artifacts

An evaluation directory contains:

```text
manifest.json
predictions.jsonl
bleu-summary.json
missing-context.jsonl
config.json
```

Each prediction stores the QA ID, dataset, language, question, all references, prediction, retrieved source IDs, reranker scores, context length, latency, availability, status, and any inference error. `config.json` records non-secret model/configuration fingerprints; API keys are never written.

## Tests

```powershell
python -m pytest
```

Tests cover CSV/JSON adapters, Unicode, canonical record counts, deduplication, multi-reference handling, missing sources, surface sparse vectors, in-memory Qdrant hybrid RRF, reranker thresholds, relative filtering, parent packing, multilingual listing detection, BLEU edge cases, and a two-QA-per-language index-to-BLEU smoke path with injected deterministic models.

## References

- DiLLeMa source components in the sibling repository: `apps/rag/nlp/doc_chunking.py`, `apps/rag/embedding/default_embedding.py`, `apps/rag/embedding/sparse_bm25.py`, `apps/rag/qdrant/client/__init__.py`, `apps/rag/llm/re_rank.py`, and `apps/app/services/retrieval_service.py`.
- [Qdrant hybrid queries and fusion](https://qdrant.tech/documentation/search/hybrid-queries/)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
- [multilingual-e5-base model card](https://huggingface.co/intfloat/multilingual-e5-base)
- [bge-reranker-v2-m3 model card](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [OpenAI Python client](https://github.com/openai/openai-python)

