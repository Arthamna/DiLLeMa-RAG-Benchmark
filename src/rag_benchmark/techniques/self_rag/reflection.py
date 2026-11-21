from __future__ import annotations

import math
from dataclasses import dataclass, field

RETRIEVAL_TOKENS = ("[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]")
RELEVANCE_TOKENS = ("[Irrelevant]", "[Relevant]")
GROUND_TOKENS = ("[Fully supported]", "[Partially supported]", "[No support / Contradictory]")
UTILITY_TOKENS = tuple(f"[Utility:{index}]" for index in range(1, 6))
UTILITY_WEIGHTS = (-1.0, -0.5, 0.0, 0.5, 1.0)
CONTROL_TOKENS = (
    "[Fully supported]", "[Partially supported]", "[No support / Contradictory]",
    "[No Retrieval]", "[Retrieval]", "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
)
MISSING_LOGPROB = -100.0


@dataclass(slots=True)
class Generation:
    text: str
    token_ids: list[int]
    step_logprobs: list[dict[int, float]]
    cumulative_logprob: float


@dataclass(frozen=True, slots=True)
class TokenIds:
    retrieval: dict[str, int]
    relevance: dict[str, int]
    ground: dict[str, int]
    utility: dict[str, int]

    @classmethod
    def from_tokenizer(cls, convert, unknown_id: int | None = None) -> "TokenIds":
        def lookup(names):
            values = {name: int(convert(name)) for name in names}
            missing = [name for name, value in values.items() if value == unknown_id or value < 0]
            if missing:
                raise ValueError(f"Self-RAG tokenizer lacks reflection tokens: {', '.join(missing)}")
            return values

        ids = cls(lookup(RETRIEVAL_TOKENS), lookup(RELEVANCE_TOKENS),
                  lookup(GROUND_TOKENS), lookup(UTILITY_TOKENS))
        if len(ids.tracked) != len(RETRIEVAL_TOKENS + RELEVANCE_TOKENS + GROUND_TOKENS + UTILITY_TOKENS):
            raise ValueError("Self-RAG reflection tokens must map to distinct ids")
        return ids

    @property
    def tracked(self) -> tuple[int, ...]:
        return tuple(sorted({*self.retrieval.values(), *self.relevance.values(),
                             *self.ground.values(), *self.utility.values()}))


@dataclass(frozen=True, slots=True)
class PassageScore:
    relevance: float
    ground: float
    utility: float
    sequence: float
    final: float
    probabilities: dict[str, dict[str, float]] = field(default_factory=dict)


def _logprob(step: dict[int, float], token_id: int) -> float:
    return float(step.get(token_id, MISSING_LOGPROB))


def retrieval_logprobs(generation: Generation, ids: TokenIds) -> dict[str, float]:
    first = generation.step_logprobs[0] if generation.step_logprobs else {}
    return {name: _logprob(first, token_id) for name, token_id in ids.retrieval.items()}


def retrieval_score(logprobs: dict[str, float]) -> float:
    retrieve, skip = logprobs["[Retrieval]"], logprobs["[No Retrieval]"]
    total = retrieve + skip
    return retrieve / total if total else float("nan")


def _probabilities_at_first(generation: Generation, token_ids: dict[str, int]) -> dict[str, float]:
    targets = set(token_ids.values())
    for index, token in enumerate(generation.token_ids):
        if token in targets:
            step = generation.step_logprobs[index]
            return {name: math.exp(_logprob(step, value)) for name, value in token_ids.items()}
    return {}


def passage_score(generation: Generation, ids: TokenIds, config) -> PassageScore:
    first = generation.step_logprobs[0] if generation.step_logprobs else {}
    relevance_probs = {name: math.exp(_logprob(first, value)) for name, value in ids.relevance.items()}
    relevance = relevance_probs["[Relevant]"] / sum(relevance_probs.values())
    ground_probs = _probabilities_at_first(generation, ids.ground) if config.use_groundness else {}
    ground = 0.0
    if len(ground_probs) == len(GROUND_TOKENS):
        total = sum(ground_probs.values())
        ground = ground_probs["[Fully supported]"] / total + 0.5 * ground_probs["[Partially supported]"] / total
    utility_probs = _probabilities_at_first(generation, ids.utility) if config.use_utility else {}
    utility = 0.0
    if len(utility_probs) == len(UTILITY_TOKENS):
        total = sum(utility_probs.values())
        utility = sum(weight * utility_probs[name] / total
                      for weight, name in zip(UTILITY_WEIGHTS, UTILITY_TOKENS))
    sequence = generation.cumulative_logprob / max(len(generation.token_ids), 1)
    final = config.w_rel * relevance + config.w_sup * ground + config.w_use * utility
    if config.use_seqscore:
        final += math.exp(sequence)
    return PassageScore(relevance, ground, utility, sequence, final, {
        "relevance": relevance_probs, "ground": ground_probs, "utility": utility_probs,
    })


def postprocess(text: str) -> str:
    for token in (*CONTROL_TOKENS, "</s>", "<|endoftext|>"):
        text = text.replace(token, "")
    text = text.replace("\n", "").strip()
    if text[:1] in {"#", ":"}:
        text = text[1:].strip()
    return text
