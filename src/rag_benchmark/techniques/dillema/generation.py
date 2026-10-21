SYSTEM_PROMPT = (
    "Answer using only the supplied evidence. Answer in the same language as the question. "
    "Return only the answer, without citations, source labels, preamble, or commentary. "
    "If the evidence is insufficient, return an empty answer."
)


def base_prompt(question: str, context: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nEvidence:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"


class DiLLeMaGenerator:
    """Compose DiLLeMa evidence prompts for a backend-neutral runtime."""

    def __init__(self, runtime):
        self.runtime = runtime

    def generate(self, question: str, context: str) -> str:
        return self.runtime.generate_prompt(base_prompt(question, context))
