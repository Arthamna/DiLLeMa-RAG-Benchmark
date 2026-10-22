from __future__ import annotations

import logging
import os

LOGGER = logging.getLogger(__name__)


def _generator(request: dict, method: str, device: str):
    from rankify.generator.generator import Generator

    config = request["generator"]
    factory_method = "basic-rag" if method == "in-context-ralm" else method
    common = {"method": factory_method, "model_name": str(config["model"])}
    if config["backend"] == "openai-compatible":
        key_env = str(config.get("api_key_env") or "LLM_API_KEY")
        common.update({
            "backend": "openai",
            "api_keys": [os.environ[key_env]],
            "base_url": str(config["base_url"]),
        })
    else:
        common.update({
            "backend": "huggingface",
            "trust_remote_code": bool(config.get("trust_remote_code", False)),
        })
        if config.get("torch_dtype"):
            common["torch_dtype"] = config["torch_dtype"] if device.startswith("cuda") else "float32"
        if config.get("attention_implementation"):
            common["attn_implementation"] = config["attention_implementation"]
        quantization = str(config.get("quantization") or "none").casefold()
        if device == "cuda":
            common["device_map"] = "auto"
            if quantization in {"4bit", "8bit"}:
                from transformers import BitsAndBytesConfig

                common["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=quantization == "4bit",
                    load_in_8bit=quantization == "8bit",
                )
        elif quantization != "none":
            LOGGER.warning("Quantization %s dinonaktifkan karena generator memakai CPU.", quantization)
    return Generator(**common)


def _generation_kwargs(request: dict) -> dict:
    maximum = int(request["generator"]["max_tokens"])
    if request["generator"]["backend"] == "openai-compatible":
        return {"temperature": 0, "max_tokens": maximum}
    return {"do_sample": False, "max_new_tokens": maximum}


def _custom_prompt(method: str):
    if method != "in-context-ralm":
        return None
    from rankify.generator.prompt_template import DEFAULT_PROMPTS, PromptTemplate

    return DEFAULT_PROMPTS[PromptTemplate.IN_CONTEXT_RALM]
