def base_prompt(question: str, context: str) -> str:
    return (
        "Answer using only the supplied evidence. Answer in the same language as the question. "
        "Return only the answer, without citations, source labels, preamble, or commentary. "
        "If the evidence is insufficient, return an empty answer."
        f"\n\nEvidence:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
