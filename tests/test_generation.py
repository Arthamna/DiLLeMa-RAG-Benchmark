from types import SimpleNamespace

from rag_benchmark.config import LLMConfig, PipelineConfig
from rag_benchmark.generation import OpenAIGenerator


def test_openai_compatible_generation_request_is_deterministic():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  deterministic answer  "))]
        )

    generator = object.__new__(OpenAIGenerator)
    generator.llm = LLMConfig("http://mock.test/v1", "not-a-real-key", "mock-model")
    generator.pipeline = PipelineConfig()
    generator.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    assert generator.generate("question", "evidence") == "deterministic answer"
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 768
    assert "references" not in captured["messages"][1]["content"].casefold()
