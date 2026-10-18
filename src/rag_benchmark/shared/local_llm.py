from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass

from .device import cuda_available

LOGGER = logging.getLogger(__name__)

RETRIEVAL_TOKEN = "<R>"
SYSTEM_PROMPT = (
    "Answer using only the supplied evidence. Answer in the same language as the question. "
    "Return only the answer, without citations, source labels, preamble, or commentary. "
    "If the evidence is insufficient, return an empty answer."
)


@dataclass(frozen=True, slots=True)
class HFModelConfig:
    model_name: str
    quantization: str = "4bit"
    trust_remote_code: bool = False
    max_input_tokens: int = 4096
    max_new_tokens: int = 768

    @classmethod
    def from_env(cls) -> "HFModelConfig":
        model_name = os.getenv("HF_MODEL_NAME", "").strip()
        if not model_name:
            raise ValueError("Missing required environment variable: HF_MODEL_NAME")
        quantization = os.getenv("HF_QUANTIZATION", "4bit").strip().casefold()
        if quantization not in {"4bit", "8bit", "none"}:
            raise ValueError("HF_QUANTIZATION must be one of: 4bit, 8bit, none")
        trust = os.getenv("HF_TRUST_REMOTE_CODE", "false").strip().casefold() in {"1", "true", "yes"}
        return cls(
            model_name=model_name,
            quantization=quantization,
            trust_remote_code=trust,
            max_input_tokens=int(os.getenv("HF_MAX_INPUT_TOKENS", "4096")),
            max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "768")),
        )

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def base_prompt(question: str, context: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nEvidence:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"


def rrag_prompt(question: str, evidence) -> str:
    documents = []
    for index, item in enumerate(evidence, 1):
        title = item.metadata.get("title") or item.metadata.get("source_path") or item.source_id
        documents.append(f"[{index}] retrieval: {RETRIEVAL_TOKEN} (Title: {title}) {item.text}")
    return base_prompt(question, "\n".join(documents))


class LocalLLM:
    """The only base-model loader/tokenizer/decoder used by every technique."""

    kind = "huggingface-local"

    def __init__(self, config: HFModelConfig, model=None, tokenizer=None):
        self.config = config
        if model is None or tokenizer is None:
            model, tokenizer = self._load(config)
        self.model = model
        self.tokenizer = tokenizer
        self._ensure_tokens()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    @staticmethod
    def _load(config: HFModelConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs = {"device_map": "auto", "trust_remote_code": config.trust_remote_code}
        if config.quantization != "none" and not cuda_available():
            LOGGER.warning(
                "CUDA GPU tidak terdeteksi; quantization %s dinonaktifkan dan LLM dimuat di CPU.",
                config.quantization,
            )
        elif config.quantization != "none":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError("Install the 'quantized' extra to use 4-bit or 8-bit loading") from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=config.quantization == "4bit",
                load_in_8bit=config.quantization == "8bit",
            )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=config.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
        return model, tokenizer

    def _ensure_tokens(self) -> None:
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        vocabulary = self.tokenizer.get_vocab()
        if RETRIEVAL_TOKEN not in vocabulary:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [RETRIEVAL_TOKEN]})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.retrieval_token_id = int(self.tokenizer.convert_tokens_to_ids(RETRIEVAL_TOKEN))

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def hidden_size(self) -> int:
        config = self.model.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return int(config.text_config.hidden_size)
        raise ValueError("Unable to determine the causal LLM hidden size")

    @property
    def fingerprint(self) -> str:
        return self.config.fingerprint()

    def tokenize(self, text: str):
        return self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.config.max_input_tokens, add_special_tokens=True,
        )

    def _decode_new_tokens(self, outputs, input_length: int) -> str:
        start = input_length if outputs.shape[1] > input_length else 0
        generated = outputs[0, start:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate(self, question: str, context: str) -> str:
        tokens = self.tokenize(base_prompt(question, context)).to(self.device)
        outputs = self.model.generate(
            **tokens, do_sample=False, max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self._decode_new_tokens(outputs, int(tokens.input_ids.shape[1]))

    def inject(self, input_ids, retrieval_embeddings):
        token_embeddings = self.model.get_input_embeddings()(input_ids)
        positions = (input_ids == self.retrieval_token_id).nonzero(as_tuple=False)
        flat = retrieval_embeddings.reshape(-1, retrieval_embeddings.shape[-1])
        if positions.shape[0] != flat.shape[0]:
            raise ValueError(
                f"Prompt has {positions.shape[0]} retrieval tokens but adapter produced {flat.shape[0]} embeddings"
            )
        updated = token_embeddings.clone()
        updated[positions[:, 0], positions[:, 1]] = flat.to(updated.dtype)
        return updated

    def generate_with_injections(self, prompt: str, retrieval_embeddings) -> str:
        tokens = self.tokenize(prompt).to(self.device)
        inputs_embeds = self.inject(tokens.input_ids, retrieval_embeddings)
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=tokens.attention_mask,
            do_sample=False,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self._decode_new_tokens(outputs, int(tokens.input_ids.shape[1]))

    def training_tensors(self, prompt: str, answer: str):
        import torch
        from transformers import BatchEncoding

        suffix = answer + (self.tokenizer.eos_token or "")
        answer_ids = self.tokenizer(suffix, add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids[: self.config.max_input_tokens - 1]
        prompt_limit = max(1, self.config.max_input_tokens - len(answer_ids))
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=True, truncation=True, max_length=prompt_limit,
        )["input_ids"]
        input_ids = torch.tensor([prompt_ids + answer_ids], device=self.device)
        combined = BatchEncoding({
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        })
        labels = input_ids.clone()
        labels[:, : len(prompt_ids)] = -100
        return combined, labels

