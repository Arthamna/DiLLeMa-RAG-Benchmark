from __future__ import annotations

from .config import LLMConfig, PipelineConfig

SYSTEM_PROMPT = """You answer questions using only the supplied evidence.
Answer in the same language as the question. Return only the answer, with no citation markers,
source labels, preamble, or extra commentary. If the evidence is insufficient, return an empty answer."""


class OpenAIGenerator:
    def __init__(self, llm: LLMConfig, pipeline: PipelineConfig | None = None):
        from openai import OpenAI

        self.llm = llm
        self.pipeline = pipeline or PipelineConfig()
        self.client = OpenAI(base_url=llm.base_url, api_key=llm.api_key)

    def generate(self, question: str, context: str) -> str:
        response = self.client.chat.completions.create(
            model=self.llm.model,
            temperature=self.pipeline.temperature,
            max_tokens=self.pipeline.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Evidence:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()

