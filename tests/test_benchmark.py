from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_benchmark.benchmark import ExperimentSpec, parse_techniques
from rag_benchmark.cli import _launch_detached_benchmark, build_parser
from rag_benchmark.techniques import TECHNIQUES, get_technique


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


def test_official_yaml_files_default_to_openai_generator(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openrouter/free")
    root = Path(__file__).resolve().parents[1]
    for dataset in ("its", "uniqa"):
        spec = ExperimentSpec.load(root / "configs" / "experiments" / f"{dataset}.yaml", "dillema")
        assert spec.generator.profile == "openai"
        assert spec.generator.backend == "openai-compatible"
        assert spec.generator.model == "openrouter/free"


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
