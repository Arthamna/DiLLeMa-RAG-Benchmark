from __future__ import annotations

import gc
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import QdrantClient

from rag_benchmark import benchmark
from rag_benchmark.benchmark import ExperimentSpec, parse_techniques
from rag_benchmark.cli import _launch_detached_benchmark
from rag_benchmark.shared.config import ChunkingConfig
from rag_benchmark.shared.datasets import DatasetBundle
from rag_benchmark.shared.metrics import read_predictions
from rag_benchmark.shared.records import ChunkRecord, DocumentRecord, QARecord, RetrievedEvidence
from rag_benchmark.techniques.retro.config import PaperEvaluationConfig
from rag_benchmark.techniques.self_rag import runner
from rag_benchmark.techniques.self_rag.config import SelfRagConfig, VARIANTS
from rag_benchmark.techniques.self_rag.pipeline import SelfRagPipeline
from rag_benchmark.techniques.self_rag.prompt import base_prompt, evidence_prompt, no_retrieval_prompt
from rag_benchmark.techniques.self_rag.reflection import (
    GROUND_TOKENS, RELEVANCE_TOKENS, RETRIEVAL_TOKENS, UTILITY_TOKENS,
    Generation, TokenIds, passage_score, postprocess, retrieval_score,
)
from rag_benchmark.techniques.self_rag.retriever import normalize_text, passage_text, prepare_text
from rag_benchmark.techniques.self_rag.store import SelfRagStore

NAMES = (*RETRIEVAL_TOKENS, *RELEVANCE_TOKENS, *GROUND_TOKENS, *UTILITY_TOKENS)
TOKEN = {name: 100 + index for index, name in enumerate(NAMES)}
IDS = TokenIds.from_tokenizer(TOKEN.__getitem__, unknown_id=0)


def log(value: float) -> float:
    return math.log(value)


def test_config_is_optional_validated_and_fingerprinted(tmp_path):
    assert SelfRagConfig.from_dict({}, tmp_path) is None
    assert SelfRagConfig.from_dict({"enabled": False}, tmp_path) is None
    parsed = SelfRagConfig.from_dict({"enabled": True}, tmp_path)
    assert parsed.model == "selfrag/selfrag_llama2_7b"
    assert parsed.retriever_model == "facebook/mcontriever-msmarco"
    assert (parsed.ndocs, parsed.threshold, parsed.w_use) == (5, .2, .5)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    local = SelfRagConfig.from_dict({"enabled": True, "model": "./snapshot"}, tmp_path)
    assert local.model == str(snapshot.resolve())
    with pytest.raises(FileNotFoundError, match="snapshot config not found"):
        SelfRagConfig.from_dict({"enabled": True, "retriever_model": "./missing"}, tmp_path)
    with pytest.raises(ValueError, match="Unknown self_rag options"):
        SelfRagConfig.from_dict({"enabled": True, "beam_width": 2}, tmp_path)
    with pytest.raises(ValueError, match="quantization"):
        SelfRagConfig.from_dict({"enabled": True, "quantization": "2bit"}, tmp_path)
    chunking = ChunkingConfig()
    index = parsed.index_fingerprint("corpus", chunking)
    assert index == replace(parsed, threshold=.5, model="other").index_fingerprint("corpus", chunking)
    assert index != replace(parsed, retriever_revision="commit").index_fingerprint("corpus", chunking)
    pipeline = parsed.pipeline_fingerprint("corpus", chunking, 100)
    assert pipeline != replace(parsed, threshold=.5).pipeline_fingerprint("corpus", chunking, 100)
    assert pipeline != replace(parsed, instruction="").pipeline_fingerprint("corpus", chunking, 100)
    assert parsed.model_fingerprint() != replace(parsed, max_new_tokens=15).model_fingerprint()


def test_prompt_matches_short_form_format_with_language_instruction():
    prompt = base_prompt("Siapa rektor?", "Answer the question in the same language as the question.")
    assert prompt == (
        "### Instruction:\nAnswer the question in the same language as the question.\n\n"
        "## Input:\n\nSiapa rektor?\n\n### Response:\n"
    )
    assert base_prompt("q", "") == "### Instruction:\nq\n\n### Response:\n"
    assert evidence_prompt("P", "Title", "Body") == "P[Retrieval]<paragraph>Title\nBody</paragraph>"
    assert no_retrieval_prompt("P") == "P[No Retrieval]"


def test_token_ids_require_every_reflection_token():
    assert len(IDS.tracked) == len(NAMES)
    with pytest.raises(ValueError, match="lacks reflection tokens"):
        TokenIds.from_tokenizer(lambda name: 0 if name == "[Relevant]" else TOKEN[name], unknown_id=0)


def test_retrieval_score_uses_original_raw_logprob_ratio():
    assert retrieval_score({"[Retrieval]": -5.0, "[No Retrieval]": -.01}) == pytest.approx(5 / 5.01)
    assert retrieval_score({"[Retrieval]": -.01, "[No Retrieval]": -5.0}) == pytest.approx(.01 / 5.01)
    assert math.isnan(retrieval_score({"[Retrieval]": 0.0, "[No Retrieval]": 0.0}))


def test_passage_score_follows_relevance_support_utility_and_sequence_rewards():
    generation = Generation(
        "[Relevant]answer[Fully supported][Utility:4]",
        [TOKEN["[Relevant]"], 7, TOKEN["[Fully supported]"], TOKEN["[Utility:4]"]],
        [
            {TOKEN["[Relevant]"]: log(.6), TOKEN["[Irrelevant]"]: log(.2)},
            {},
            {TOKEN["[Fully supported]"]: log(.5), TOKEN["[Partially supported]"]: log(.3),
             TOKEN["[No support / Contradictory]"]: log(.2)},
            {TOKEN[name]: log(value) for name, value in zip(UTILITY_TOKENS, (.1, .1, .2, .4, .2))},
        ],
        -2.0,
    )
    score = passage_score(generation, IDS, SelfRagConfig())
    assert score.relevance == pytest.approx(.75)
    assert score.ground == pytest.approx(.65)
    assert score.utility == pytest.approx(.25)
    assert score.sequence == pytest.approx(-.5)
    assert score.final == pytest.approx(.75 + .65 + .5 * .25 + math.exp(-.5))
    plain = passage_score(generation, IDS, replace(
        SelfRagConfig(), use_groundness=False, use_utility=False, use_seqscore=False,
    ))
    assert (plain.ground, plain.utility, plain.final) == (0.0, 0.0, pytest.approx(.75))


def test_postprocess_removes_reflection_tokens_and_leading_markers():
    assert postprocess("[Relevant]Jawaban\nlengkap[Fully supported][Utility:5]</s>") == "Jawabanlengkap"
    assert postprocess(": risposta") == "risposta"
    assert postprocess("") == ""


def test_mcontriever_text_preprocessing():
    assert normalize_text("a–b “q” it’s…") == 'a-b "q" it\'s...'
    config = SelfRagConfig()
    assert prepare_text("Ab—C", config) == "ab-c"
    assert prepare_text("Ab", replace(config, lowercase=False, normalize_text=False)) == "Ab"
    assert passage_text("Title", "Body") == "Title Body"
    assert passage_text("", "Body") == "Body"


def _chunk(identifier: str, text: str) -> ChunkRecord:
    return ChunkRecord(identifier, identifier, "its", "id", text, text, text, identifier,
                       f"{identifier}.txt", identifier.title())


def test_store_uses_dot_product_search(tmp_path):
    client = QdrantClient(location=":memory:")
    store = SelfRagStore(tmp_path, "test", client=client)
    store.recreate("id", 2)
    store.add("id", [_chunk("a", "alpha"), _chunk("b", "beta")], np.asarray([[2., 0.], [0., 1.]]))
    hits = store.search("id", np.asarray([1., 1.]), 2)
    assert [hit.chunk_id for hit in hits] == ["a", "b"]
    assert hits[0].metadata["title"] == "A"
    assert hits[0].rerank_score == pytest.approx(2.0)
    client.close()


class FakeLLM:
    revision = "selfrag-commit"
    quantization = "none"
    dtype = "float32"

    def __init__(self, *args, events=None):
        self.ids = IDS
        self.prompts = []
        self.events = events
        if events is not None:
            events.append("load-self-rag")

    def generate(self, prompts, max_new_tokens=None):
        result = []
        for prompt in prompts:
            self.prompts.append((prompt, max_new_tokens))
            if prompt.endswith("[No Retrieval]"):
                result.append(Generation("closed book answer", [5, 2], [{}, {}], -1.0))
            elif "<paragraph>" not in prompt:
                retrieve = "retrieve" in prompt
                result.append(Generation("", [TOKEN["[Retrieval]"]], [{
                    TOKEN["[Retrieval]"]: -5.0 if retrieve else -.01,
                    TOKEN["[No Retrieval]"]: -.01 if retrieve else -5.0,
                }], -.01))
            elif "alpha" in prompt.split("<paragraph>")[1]:
                result.append(self._passage("alpha answer", .9, .8, "[Utility:5]"))
            else:
                result.append(self._passage("beta answer", .1, .1, "[Utility:1]"))
        return result

    @staticmethod
    def _passage(answer, relevant, supported, utility):
        utility_step = {TOKEN[name]: log(.9 if name == utility else .025) for name in UTILITY_TOKENS}
        return Generation(
            f"[Relevant]{answer}[Fully supported]{utility}",
            [TOKEN["[Relevant]"], 9, TOKEN["[Fully supported]"], TOKEN[utility]],
            [
                {TOKEN["[Relevant]"]: log(relevant), TOKEN["[Irrelevant]"]: log(1 - relevant)},
                {},
                {TOKEN["[Fully supported]"]: log(supported),
                 TOKEN["[Partially supported]"]: log((1 - supported) / 2),
                 TOKEN["[No support / Contradictory]"]: log((1 - supported) / 2)},
                utility_step,
            ],
            -1.0,
        )


def _evidence(identifier: str) -> RetrievedEvidence:
    return RetrievedEvidence(identifier, identifier, f"{identifier} text", 1.0, {"title": identifier})


def test_pipeline_shares_passage_generations_between_variants():
    llm = FakeLLM()
    pipeline = SelfRagPipeline(llm, replace(SelfRagConfig(), ndocs=2))
    outcomes = pipeline.run("please retrieve", [_evidence("beta"), _evidence("alpha"), _evidence("gamma")])
    assert set(outcomes) == set(VARIANTS)
    for name in VARIANTS:
        assert outcomes[name].prediction == "alpha answer"
        assert outcomes[name].retrieval_used is True
        assert [item.chunk_id for item in outcomes[name].evidence] == ["alpha", "beta"]
        assert outcomes[name].context == "alpha text"
    assert outcomes["self_rag_adaptive"].diagnostics["retrieve"] is True
    assert "retrieve" not in outcomes["self_rag_always"].diagnostics
    passage_prompts = [prompt for prompt, _ in llm.prompts if "<paragraph>" in prompt]
    assert len(passage_prompts) == 2
    assert [tokens for prompt, tokens in llm.prompts if "<paragraph>" not in prompt] == [1]


def test_adaptive_without_retrieval_generates_from_no_retrieval_prompt():
    llm = FakeLLM()
    pipeline = SelfRagPipeline(llm, SelfRagConfig())
    outcome = pipeline.run("plain question", [_evidence("alpha")], ("self_rag_adaptive",))["self_rag_adaptive"]
    assert outcome.prediction == "closed book answer"
    assert outcome.retrieval_used is False
    assert outcome.evidence == []
    assert outcome.diagnostics["retrieval_score"] == pytest.approx(.01 / 5.01)
    assert not any("<paragraph>" in prompt for prompt, _ in llm.prompts)
    with pytest.raises(ValueError, match="Unsupported"):
        pipeline.run("q", [], ("self_rag_long_form",))


@pytest.fixture
def self_rag_environment(tmp_path, monkeypatch):
    client = QdrantClient(location=":memory:")
    events, llms = [], []

    class Encoder:
        vector_size = 2
        revision = "mcontriever-commit"

        def __init__(self, *args):
            events.append("load-mcontriever")

        @staticmethod
        def _vector(text):
            alpha = 1.0 if "alpha" in text.casefold() else 0.0
            return np.asarray([alpha, 1.0 - alpha])

        def encode_passages(self, texts):
            return np.asarray([self._vector(text) for text in texts])

        def encode_query(self, text):
            return self._vector(text)

        def __del__(self):
            events.append("unload-mcontriever")

    class Store(SelfRagStore):
        def __init__(self, root, namespace):
            super().__init__(root, namespace, client=client)

    def llm_factory(*args):
        llms.append(FakeLLM(events=events))
        return llms[-1]

    monkeypatch.setattr(runner, "MContriever", Encoder)
    monkeypatch.setattr(runner, "SelfRagStore", Store)
    monkeypatch.setattr(runner, "SelfRagLLM", llm_factory)
    monkeypatch.setattr(runner, "resolve_torch_device", lambda *args: "cpu")
    monkeypatch.setattr(runner, "_release_gpu", gc.collect)
    yield tmp_path, events, llms
    client.close()


def spec_and_bundle(tmp_path):
    documents = [
        DocumentRecord("alpha", "its", "id", "Alpha", "alpha " * 80, "alpha.txt"),
        DocumentRecord("beta", "its", "id", "Beta", "beta " * 80, "beta.txt"),
    ]
    questions = [
        QARecord("q1", "its", "id", "alpha retrieve question", ["alpha answer"], "test", source_ids=["alpha"]),
        QARecord("q2", "its", "id", "beta question", ["beta answer"], "test", source_ids=["beta"]),
        QARecord("q3", "its", "id", "missing", ["x"], "test", available=False),
    ]
    spec = ExperimentSpec(
        dataset="its", source=tmp_path, index_root=tmp_path / "indexes",
        candidates=2, top_k=1, evaluation_split="test", generator=None,
        retro=None, piperag=None, services=None, paper=PaperEvaluationConfig(),
        self_rag=replace(SelfRagConfig(), ndocs=2), techniques=("self_rag",),
    )
    return spec, DatasetBundle(documents, questions)


def test_runner_writes_two_isolated_outputs_and_ablation(self_rag_environment):
    tmp_path, events, llms = self_rag_environment
    spec, bundle = spec_and_bundle(tmp_path)
    manifest = runner.run_index(spec, bundle)
    reports = runner.run_evaluate(spec, bundle, tmp_path / "run")
    assert manifest["retriever_revision"] == "mcontriever-commit"
    assert set(reports) == set(VARIANTS)
    rows = {name: read_predictions(tmp_path / "run" / name / "predictions.jsonl") for name in VARIANTS}
    adaptive, always = rows["self_rag_adaptive"], rows["self_rag_always"]
    assert [row["id"] for row in adaptive] == ["q1", "q2", "q3"]
    assert [row["prediction"] for row in always[:2]] == ["alpha answer", "alpha answer"]
    assert [row["prediction"] for row in adaptive[:2]] == ["alpha answer", "closed book answer"]
    assert adaptive[1]["status"] == "no_evidence"
    assert adaptive[2]["status"] == always[2]["status"] == "missing_context"
    assert always[0]["retrieved_source_ids"] == ["alpha", "beta"]
    assert always[0]["diagnostics"]["candidate_relevant_ranks"] == [1]
    assert always[0]["decoding_config"] == spec.self_rag.decoding_dict()
    assert always[0]["model_fingerprint"] == spec.self_rag.model_fingerprint()
    assert always[0]["latency_ms"] >= always[0]["diagnostics"]["retrieval_latency_ms"]
    passage_prompts = [prompt for prompt, _ in llms[0].prompts if "<paragraph>" in prompt]
    assert len(passage_prompts) == 4
    assert all("alpha answer" not in prompt for prompt, _ in llms[0].prompts)
    ablation = json.loads((tmp_path / "run" / "self-rag-ablation.json").read_text())
    assert ablation["adaptive_retrieval_rate"] == {"id": .5}
    assert set(ablation["adaptive_minus_always"]) == {"id"}
    assert events.index("unload-mcontriever") < events.index("load-self-rag")
    benchmark._write_leaderboard(tmp_path / "run", reports)
    leaderboard = json.loads((tmp_path / "run" / "leaderboard.json").read_text())
    assert {row["technique"] for row in leaderboard} == set(VARIANTS)


def test_runner_rejects_missing_index(self_rag_environment):
    tmp_path, _, _ = self_rag_environment
    spec, bundle = spec_and_bundle(tmp_path)
    with pytest.raises(FileNotFoundError, match="Self-RAG index missing"):
        runner.run_evaluate(spec, bundle, tmp_path / "run")


@pytest.mark.parametrize("operation", ["index", "evaluate"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_each_variant_routes_through_cli(tmp_path, monkeypatch, capsys, operation, variant):
    from rag_benchmark import cli

    spec, bundle = spec_and_bundle(tmp_path)
    loads = []
    monkeypatch.setattr(ExperimentSpec, "load", lambda path, *args: loads.append(args) or spec)
    monkeypatch.setattr("rag_benchmark.shared.datasets.load_dataset", lambda *args: bundle)
    calls = []
    monkeypatch.setattr(runner, "run_index", lambda loaded, data: calls.append(("index",)) or {"ok": True})
    monkeypatch.setattr(
        runner, "run_evaluate",
        lambda loaded, data, output, variants: calls.append(("evaluate", variants)) or {"ok": True},
    )
    arguments = [operation, "--technique", variant, "--dataset", "its", "--config", "experiment.yaml"]
    arguments += ["--source", str(tmp_path)] if operation == "index" else ["--output", str(tmp_path / "run")]
    cli.main(arguments)
    assert loads == [("self_rag",)]
    assert calls[0][0] == operation
    if operation == "evaluate":
        assert calls[0][1] == (variant,)
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def _self_rag_only_yaml(tmp_path, enabled=True):
    (tmp_path / "dataset").mkdir(exist_ok=True)
    config = tmp_path / "experiment.yaml"
    config.write_text("\n".join([
        "dataset: its", "source: dataset", "index_root: indexes",
        "retrieval:", "  candidates: 10", "  top_k: 5", "  rrf_k: 60",
        "self_rag:", f"  enabled: {str(enabled).lower()}",
    ]), encoding="utf-8")
    return config


def test_experiment_selects_self_rag_without_shared_generator_or_rankify(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_MODEL_NAME", raising=False)
    config = _self_rag_only_yaml(tmp_path)
    spec = ExperimentSpec.load(config, "self_rag")
    assert spec.techniques == ("self_rag",)
    assert spec.generator is None
    assert spec.self_rag.ndocs == 5
    assert spec.public_dict()["techniques"] == ["self_rag"]
    with pytest.raises(ValueError, match="generator.backend"):
        ExperimentSpec.load(config)
    with pytest.raises(ValueError, match="not enabled"):
        ExperimentSpec.load(_self_rag_only_yaml(tmp_path, enabled=False), "self_rag")
    assert parse_techniques("rank_zephyr, self_rag") == ("self_rag", "rank_zephyr")
    with pytest.raises(ValueError, match="subset"):
        parse_techniques("self_rag,unknown")


def test_benchmark_runs_self_rag_alone(tmp_path, monkeypatch):
    config = _self_rag_only_yaml(tmp_path)
    _, bundle = spec_and_bundle(tmp_path)
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("other techniques must not run")

    monkeypatch.setattr(benchmark, "load_dataset", lambda *args: bundle)
    monkeypatch.setattr(benchmark, "index_dillema", forbidden)
    monkeypatch.setattr(benchmark, "evaluate_dillema", forbidden)
    monkeypatch.setattr(runner, "run_index", lambda spec, data: calls.append("index") or {"status": "complete"})
    monkeypatch.setattr(runner, "run_evaluate", lambda spec, data, output: calls.append("evaluate") or {})
    manifest = benchmark.run_benchmark(config, "all", tmp_path / "run", "self_rag")
    assert calls == ["index", "evaluate"]
    assert manifest["config"]["services"] is None
    assert manifest["result"]["techniques"] == ["self_rag"]
    assert manifest["result"]["index"] == {"self_rag": {"status": "complete"}}


def test_detached_benchmark_forwards_technique_selection(tmp_path, monkeypatch):
    monkeypatch.setattr("rag_benchmark.cli.platform.system", lambda: "Linux")
    loads = []
    monkeypatch.setattr(ExperimentSpec, "load", lambda path, *args: loads.append(args) or SimpleNamespace())
    observed = {}
    monkeypatch.setattr(
        "rag_benchmark.cli.subprocess.Popen",
        lambda command, **kwargs: observed.update(command=command) or SimpleNamespace(pid=1),
    )
    _launch_detached_benchmark(tmp_path / "experiment.yaml", "all", tmp_path / "run", "self_rag")
    assert loads == [("self_rag",)]
    assert observed["command"][-2:] == ["--techniques", "self_rag"]
