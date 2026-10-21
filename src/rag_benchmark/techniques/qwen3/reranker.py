from __future__ import annotations

import logging

from ...shared.device import synchronize_cuda
from .embedding import model_options

LOGGER = logging.getLogger(__name__)
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def relevance_probability(logits, no_id: int, yes_id: int):
    return logits[:, [no_id, yes_id]].float().softmax(dim=1)[:, 1]


def format_pair(instruction: str, question: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {question}\n<Document>: {document}"


class Qwen3Reranker:
    def __init__(self, config, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.config = config
        self.device = device
        options = model_options(config.reranker_model, config.reranker_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.reranker_model, padding_side="left", trust_remote_code=False, **options,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {
            "torch_dtype": getattr(torch, config.torch_dtype) if device.startswith("cuda") else torch.float32,
            "attn_implementation": config.attention_implementation if device.startswith("cuda") else "sdpa",
            "trust_remote_code": False, **options,
        }
        self.quantization = config.reranker_quantization if device.startswith("cuda") else "none"
        if self.quantization != "none":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.quantization == "4bit", load_in_8bit=self.quantization == "8bit",
                bnb_4bit_compute_dtype=getattr(torch, config.torch_dtype),
            )
            kwargs["device_map"] = {"": device}
        elif config.reranker_quantization != "none":
            LOGGER.warning("Qwen3 CPU fallback: quantization disabled, using float32")
        self.model = AutoModelForCausalLM.from_pretrained(config.reranker_model, **kwargs).eval()
        if self.quantization == "none":
            self.model.to(device)
        self.revision = getattr(self.model.config, "_commit_hash", None) or config.reranker_revision
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.no_id is None or self.yes_id is None or self.no_id == self.yes_id:
            raise ValueError("Qwen3 reranker tokenizer must have distinct yes/no tokens")
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        if config.reranker_max_length <= len(self.prefix) + len(self.suffix):
            raise ValueError("Qwen3 reranker_max_length cannot fit the scoring template")

    def score(self, question: str, passages) -> list[float]:
        import torch

        scores = []
        for offset in range(0, len(passages), self.config.reranker_batch_size):
            texts = [format_pair(self.config.instruction, question, passage)
                     for passage in passages[offset:offset + self.config.reranker_batch_size]]
            tokens = self.tokenizer(
                texts, padding=False, truncation=True, add_special_tokens=False,
                max_length=self.config.reranker_max_length - len(self.prefix) - len(self.suffix),
                return_attention_mask=False,
            )
            tokens["input_ids"] = [self.prefix + ids + self.suffix for ids in tokens["input_ids"]]
            inputs = self.tokenizer.pad(tokens, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                # Only compute the final vocabulary logits; full sequence logits waste VRAM.
                output = self.model(**inputs, use_cache=False, logits_to_keep=1)
                probabilities = relevance_probability(output.logits[:, -1], self.no_id, self.yes_id)
            synchronize_cuda(self.device, "qwen3-reranking")
            scores.extend(probabilities.cpu().tolist())
        return scores
