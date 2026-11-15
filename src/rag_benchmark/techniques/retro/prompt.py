from __future__ import annotations


def build_qa_prompt_ids(tokenizer, question: str, chunk_size: int = 64) -> list[int]:
    prefix = tokenizer.encode("Question: ", add_special_tokens=False)
    suffix = tokenizer.encode("\nAnswer: ", add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)
    available = chunk_size - len(prefix) - len(suffix)
    if available <= 0:
        raise ValueError("Prompt framing does not fit in the RETRO chunk")
    question_ids = question_ids[-available:]
    ids = [*prefix, *question_ids, *suffix]
    pad_id = int(tokenizer.pad_token_id or 0)
    return [pad_id] * (chunk_size - len(ids)) + ids


def extract_answer(decoded: str) -> str:
    marker = "Answer:"
    return decoded.rsplit(marker, 1)[-1].strip() if marker in decoded else decoded.strip()
