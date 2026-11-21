from __future__ import annotations

RETRIEVAL = "[Retrieval]"
NO_RETRIEVAL = "[No Retrieval]"


def instruction_text(question: str, instruction: str) -> str:
    return f"{instruction}\n\n## Input:\n\n{question}" if instruction else question


def base_prompt(question: str, instruction: str) -> str:
    return f"### Instruction:\n{instruction_text(question, instruction)}\n\n### Response:\n"


def evidence_prompt(prompt: str, title: str, text: str) -> str:
    return prompt + f"{RETRIEVAL}<paragraph>{title}\n{text}</paragraph>"


def no_retrieval_prompt(prompt: str) -> str:
    return prompt + NO_RETRIEVAL
