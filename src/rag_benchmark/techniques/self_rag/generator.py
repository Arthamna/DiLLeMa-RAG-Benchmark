from __future__ import annotations

from pathlib import Path

from ...shared.device import synchronize_cuda
from .reflection import Generation, TokenIds


def model_options(model: str, revision: str) -> dict:
    return {"local_files_only": True} if Path(model).is_dir() else {"revision": revision}


class SelfRagLLM:
    def __init__(self, config, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = device
        options = model_options(config.model, config.revision)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, padding_side="left", **options)
        if self.tokenizer.pad_token_id is None:
            vocabulary = self.tokenizer.get_vocab()
            self.tokenizer.pad_token = "<pad>" if "<pad>" in vocabulary else self.tokenizer.eos_token
        cuda = device.startswith("cuda")
        dtype = getattr(torch, config.torch_dtype) if cuda else torch.float32
        kwargs = {"torch_dtype": dtype}
        self.quantization = config.quantization if cuda else "none"
        if self.quantization != "none":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.quantization == "4bit",
                load_in_8bit=self.quantization == "8bit",
                bnb_4bit_compute_dtype=dtype,
            )
            kwargs["device_map"] = {"": device}
        self.model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs, **options)
        if self.quantization == "none":
            self.model.to(device)
        self.model.eval()
        self.dtype = config.torch_dtype if cuda else "float32"
        self.revision = getattr(self.model.config, "_commit_hash", None) or config.revision
        self.ids = TokenIds.from_tokenizer(
            self.tokenizer.convert_tokens_to_ids, self.tokenizer.unk_token_id,
        )

    def generate(self, prompts: list[str], max_new_tokens: int | None = None) -> list[Generation]:
        size = self.config.generation_batch_size
        results: list[Generation] = []
        for offset in range(0, len(prompts), size):
            results.extend(self._generate(prompts[offset:offset + size],
                                          max_new_tokens or self.config.max_new_tokens))
        return results

    def _generate(self, prompts: list[str], max_new_tokens: int) -> list[Generation]:
        import torch
        from transformers import GenerationConfig

        tokens = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_length = int(tokens["input_ids"].shape[1])
        if input_length > self.config.max_input_tokens:
            raise ValueError(
                f"Self-RAG prompt has {input_length} tokens; max_input_tokens is {self.config.max_input_tokens}"
            )
        tokens = tokens.to(self.device)
        eos = self.tokenizer.eos_token_id
        settings = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=eos,
            bos_token_id=self.tokenizer.bos_token_id,
            output_logits=True, return_dict_in_generate=True,
        )
        with torch.inference_mode():
            output = self.model.generate(**tokens, generation_config=settings)
            generated = output.sequences[:, input_length:]
            logprobs = torch.log_softmax(torch.stack(output.logits, dim=1).float(), dim=-1)
            tracked = list(self.ids.tracked)
            index = torch.tensor(tracked, device=logprobs.device)
            tracked_values = logprobs.index_select(-1, index).cpu().tolist()
            chosen = logprobs.gather(-1, generated.unsqueeze(-1)).squeeze(-1).cpu().tolist()
        synchronize_cuda(self.device, "self-rag-generation")
        results = []
        for row, sequence in enumerate(generated.cpu().tolist()):
            length = sequence.index(eos) + 1 if eos in sequence else len(sequence)
            token_ids = sequence[:length]
            results.append(Generation(
                text=self.tokenizer.decode(
                    [token for token in token_ids if token != eos], skip_special_tokens=False,
                ),
                token_ids=token_ids,
                step_logprobs=[dict(zip(tracked, values)) for values in tracked_values[row][:length]],
                cumulative_logprob=float(sum(chosen[row][:length])),
            ))
        return results
