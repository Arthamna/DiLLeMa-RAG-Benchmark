# Third-party notices

## RETRO

Model architecture and evaluation conventions originate from
[TobiasNorlund/retro](https://github.com/TobiasNorlund/retro), commit `6945cdd`,
licensed under Apache License 2.0. Model weights are supplied separately and
are not redistributed by this repository.

## PipeRAG

The ONNX inference model, stale retrieval schedule, and performance-model
conventions are derived from
[amazon-science/piperag](https://github.com/amazon-science/piperag), commit
`b6eb158`, licensed under Apache License 2.0. The minimum inference source and
upstream license are vendored under `src/rag_benchmark/_vendor/piperag/`.

PipeRAG citation: *PipeRAG: Fast Retrieval-Augmented Generation via Algorithm-
System Co-design* (Amazon Science). No PipeRAG or RETRO checkpoint is included.

## Self-RAG and RankZephyr

The Self-RAG technique reimplements the short-form inference procedure
(prompt format, reflection-token scoring, and adaptive retrieval threshold) of
[Self-RAG](https://github.com/AkariAsai/self-rag) by Asai et al., distributed
under the MIT License. The
[selfrag_llama2_7b](https://huggingface.co/selfrag/selfrag_llama2_7b) weights are
derived from Llama 2 and downloaded separately under the Llama 2 Community
License; they are not redistributed.

Retrieval uses [mContriever-msmarco](https://huggingface.co/facebook/mcontriever-msmarco)
from [Contriever](https://github.com/facebookresearch/contriever), released under
CC BY-NC 4.0 (non-commercial use only). Its text normalization is adapted from
ChemDataExtractor (MIT License, Copyright 2016 Matt Swain).

Asai et al. (2024), *Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection*, https://arxiv.org/abs/2310.11511.

Izacard et al. (2022), *Unsupervised Dense Information Retrieval with
Contrastive Learning*, https://arxiv.org/abs/2112.09118.

The RankZephyr worker uses [RankLLM](https://github.com/castorini/rank_llm)
under the [Apache License 2.0](https://github.com/castorini/rank_llm/blob/0fd5678ffc43488531ba26a42f367d42480a75ec/LICENSE).
The [RankZephyr 7B v1 full checkpoint](https://huggingface.co/castorini/rank_zephyr_7b_v1_full)
is downloaded separately under the MIT License. Pradeep et al. (2023),
*RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!*
[Paper](https://arxiv.org/abs/2312.02724).
