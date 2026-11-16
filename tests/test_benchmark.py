from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_benchmark.benchmark import ExperimentSpec, _gpu_lease, parse_techniques
from rag_benchmark.cli import _launch_detached_benchmark, build_parser
from rag_benchmark.shared.datasets import load_dataset
from rag_benchmark.shared.local_llm import HFModelConfig
from rag_benchmark.shared.records import QARecord, TechniqueResult
from rag_benchmark.techniques import TECHNIQUES, get_technique
from rag_benchmark.techniques.dillema.augmentation import PackedContext
from rag_benchmark.techniques.dillema.runner import run_evaluate as evaluate_dillema


def _config(tmp_path: Path, techniques="[retro, piperag]") -> Path:
    source = tmp_path / "dataset"
    source.mkdir(exist_ok=True)
    path = tmp_path / "experiment.yaml"
    path.write_text(
        f"""dataset: its
source: dataset
index_root: indexes
techniques: {techniques}
evaluation: {{split: test}}
services:
  retrieval_endpoint: localhost:50051
  inference_endpoint: localhost:50052
retro:
  checkpoint: /models/model.ckpt
  model_config: /models/retro.json
  onnx_encoder: /onnx/encoder.onnx
  onnx_decoder: /onnx/decoder.onnx
piperag:
  retrieval_interval: 64
  stale_steps: 64
paper_evaluation:
  intervals: [8, 16, 32, 64]
  nprobes: [1, 2, 4]
""",
        encoding="utf-8",
    )
    return path


def test_registry_and_cli_contain_exactly_three_techniques():
    assert tuple(TECHNIQUES) == ("dillema", "retro", "piperag")
    assert get_technique("retro").import_path.endswith("RetroPipeline")
    with pytest.raises(ValueError, match="Unsupported"):
        get_technique("rankify")
    parser = build_parser()
    assert parser.parse_args([
        "benchmark", "--config", "x.yaml", "--stage", "all", "--output", "out",
        "--techniques", "dillema,retro,piperag",
    ]).stage == "all"
    assert parser.parse_args(["doctor", "--config", "x.yaml"]).command == "doctor"


def test_benchmark_parser_accepts_detach_aliases():
    parser = build_parser()
    arguments = [
        "benchmark", "--config", "experiment.yaml", "--stage", "all",
        "--output", "artifacts/benchmarks/run",
    ]
    assert parser.parse_args(arguments).detach is False
    assert parser.parse_args([*arguments, "-d"]).detach is True
    assert parser.parse_args([*arguments, "--detach"]).detach is True


def test_technique_parser_is_ordered_and_rejects_legacy_names():
    assert parse_techniques("piperag,dillema") == ("dillema", "piperag")
    with pytest.raises(ValueError, match="dillema, retro, piperag"):
        parse_techniques("rankify")


def test_experiment_yaml_loads_typed_remote_configuration(tmp_path):
    spec = ExperimentSpec.load(_config(tmp_path))
    assert spec.techniques == ("retro", "piperag")
    assert spec.retro.chunk_size == 64
    assert spec.retro.max_new_tokens == 128
    assert spec.piperag.stale_steps == 64
    assert spec.services.retrieval_index_root == "/data/indexes"
    assert spec.paper.seed == 42


def test_official_yaml_files_parse_for_remote_pair():
    root = Path(__file__).resolve().parents[1]
    for dataset in ("its", "uniqa"):
        spec = ExperimentSpec.load(root / "configs" / "experiments" / f"{dataset}.yaml", "retro,piperag")
        assert spec.dataset == dataset
        assert spec.techniques == ("retro", "piperag")


def test_official_its_yaml_points_to_parseable_corpus_root():
    root = Path(__file__).resolve().parents[1]
    spec = ExperimentSpec.load(root / "configs" / "experiments" / "its.yaml", "retro,piperag")

    bundle = load_dataset(spec.dataset, spec.source)

    assert spec.source.name == "dataset"
    assert bundle.documents
    assert bundle.questions


def test_official_yaml_files_load_selected_generator_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openrouter/free")
    snapshot = tmp_path / "qwen-snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HF_MODEL_NAME", str(snapshot.resolve()))
    root = Path(__file__).resolve().parents[1]
    for dataset in ("its", "uniqa"):
        spec = ExperimentSpec.load(root / "configs" / "experiments" / f"{dataset}.yaml", "dillema")
        assert spec.generator.profile in {"openai", "qwen"}
        if spec.generator.profile == "openai":
            assert spec.generator.backend == "openai-compatible"
            assert spec.generator.model == "openrouter/free"
        else:
            assert spec.generator.backend == "huggingface"
            assert spec.generator.model == str(snapshot.resolve())


def test_experiment_yaml_can_toggle_local_qwen_profile(tmp_path, monkeypatch):
    source = tmp_path / "dataset"
    source.mkdir()
    snapshot = tmp_path / "qwen-snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HF_MODEL_NAME", str(snapshot.resolve()))
    config = tmp_path / "qwen.yaml"
    config.write_text(
        """dataset: its
source: dataset
techniques: [dillema]
generator:
  active: qwen
  profiles:
    openai:
      backend: openai-compatible
      model_env: LLM_MODEL
      temperature: 0
      max_tokens: 128
    qwen:
      backend: huggingface
      model_env: HF_MODEL_NAME
      temperature: 0
      max_tokens: 128
""",
        encoding="utf-8",
    )

    spec = ExperimentSpec.load(config)

    assert spec.generator.profile == "qwen"
    assert spec.generator.backend == "huggingface"
    assert spec.generator.model == str(snapshot.resolve())


def test_detached_benchmark_launches_same_cli_and_tracks_pid(tmp_path, monkeypatch):
    config = _config(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr("rag_benchmark.cli.platform.system", lambda: "Linux")
    observed = {}

    def fake_popen(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr("rag_benchmark.cli.subprocess.Popen", fake_popen)
    result = _launch_detached_benchmark(config, "all", output, "retro,piperag")
    assert result["pid"] == 4321
    assert observed["command"] == [
        sys.executable, "-u", "-m", "rag_benchmark", "benchmark", "--config", str(config.resolve()),
        "--stage", "all", "--output", str(output.resolve()), "--techniques", "retro,piperag",
    ]
    assert observed["kwargs"]["stdin"] == subprocess.DEVNULL
    assert observed["kwargs"]["start_new_session"] is True
    assert (output / "benchmark.pid").read_text() == "4321\n"


def test_detached_benchmark_rejects_live_pid(tmp_path, monkeypatch):
    config = _config(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "benchmark.pid").write_text("123\n", encoding="utf-8")
    monkeypatch.setattr("rag_benchmark.cli.platform.system", lambda: "Linux")
    monkeypatch.setattr("rag_benchmark.cli._pid_is_running", lambda pid: pid == 123)
    monkeypatch.setattr(
        "rag_benchmark.cli.subprocess.Popen",
        lambda *args, **kwargs: pytest.fail("Popen must not run while the recorded PID is alive"),
    )

    with pytest.raises(RuntimeError, match="already running with PID 123"):
        _launch_detached_benchmark(config, "all", output, "retro,piperag")


def test_gpu_lease_rejects_parallel_runs_even_with_different_outputs(tmp_path):
    root = tmp_path / "artifacts"
    with _gpu_lease(root, tmp_path / "run-a"):
        with pytest.raises(RuntimeError, match="GPU benchmark is already active"):
            with _gpu_lease(root, tmp_path / "run-b"):
                pytest.fail("second benchmark must not acquire the shared GPU lease")


def test_dillema_stages_retrieval_before_loading_local_generator(tmp_path, monkeypatch):
    events = []
    question = QARecord("q1", "its", "id", "question", ["answer"], "test")
    bundle = SimpleNamespace(documents=[], questions_for_split=lambda _split: [question])

    class Store:
        def __init__(self, _path):
            events.append("store")

        def close(self):
            events.append("store-closed")

    class Pipeline:
        def __init__(self, _store, _embedder, _reranker, _config, generator=None):
            self.generator = generator

        def retrieve(self, _collection, _question):
            events.append("retrieve")
            return PackedContext("evidence", [], {
                "retrieval_latency_ms": 1.0, "rerank_latency_ms": 2.0,
            })

        def answer_packed(self, prompt, packed):
            events.append("generate")
            return TechniqueResult(
                self.generator.generate(prompt, packed.text), packed.text, packed.evidence,
                {**packed.diagnostics, "generation_latency_ms": 3.0},
            )

    class Runtime:
        kind = "huggingface-local"
        fingerprint = "runtime"
        config = HFModelConfig("local-model", max_new_tokens=128)

        @staticmethod
        def generate_prompt(_prompt):
            return "answer"

    def runtime_factory():
        events.append("runtime-loaded")
        assert events.index("retrieval-released") < events.index("runtime-loaded")
        return Runtime()

    monkeypatch.setattr(
        "rag_benchmark.techniques.dillema.runner._load_index_manifest",
        lambda *_: ({"corpus_fingerprint": "corpus"}, tmp_path / "source"),
    )
    monkeypatch.setattr("rag_benchmark.techniques.dillema.runner.load_dataset", lambda *_: bundle)
    monkeypatch.setattr("rag_benchmark.techniques.dillema.runner.resolve_torch_device", lambda *_: "cuda")
    monkeypatch.setattr("rag_benchmark.techniques.dillema.runner.QdrantStore", Store)
    monkeypatch.setattr("rag_benchmark.techniques.dillema.runner.E5Embedding", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("rag_benchmark.techniques.dillema.runner.CrossEncoderReranker", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("rag_benchmark.techniques.dillema.runner.RAGPipeline", Pipeline)
    monkeypatch.setattr(
        "rag_benchmark.techniques.dillema.runner.release_cuda_memory",
        lambda stage: events.append("retrieval-released" if "retrieval" in stage else "generation-released"),
    )

    evaluate_dillema(
        "dillema", "its", tmp_path / "index", tmp_path / "output",
        runtime_factory=runtime_factory, model_fingerprint="model",
        decoding_config={"max_tokens": 128},
    )

    assert events.index("retrieve") < events.index("retrieval-released") < events.index("runtime-loaded")
    assert events.index("runtime-loaded") < events.index("generate") < events.index("generation-released")
