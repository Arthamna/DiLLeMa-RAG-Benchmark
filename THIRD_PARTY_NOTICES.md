# Third-party notices

The optional isolated worker integrates
[Rankify](https://github.com/DataScienceUIBK/Rankify):

> Rankify: A Comprehensive Python Toolkit for Retrieval, Re-Ranking, and RAG

Rankify is distributed under the Apache License 2.0. Copyright remains with its
original authors. This repository does not redistribute Rankify model weights.

The Qwen3 technique follows the pooling, instruction formatting, and yes/no
logit scoring examples from [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)
by the Qwen team, distributed under the Apache License 2.0. The
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and
[Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) weights are
downloaded separately under their upstream licenses and are not redistributed.

Zhang et al. (2025), *Qwen3 Embedding: Advancing Text Embedding and Reranking
Through Foundation Models*, https://arxiv.org/abs/2506.05176.

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
