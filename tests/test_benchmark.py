import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from rag_benchmark.benchmark import ExperimentSpec
from rag_benchmark.cli import _launch_detached_benchmark, build_parser
from rag_benchmark.shared.comparison import run_compare_techniques
from rag_benchmark.shared.config import GenerationSpec
from rag_benchmark.shared.metrics import write_jsonl
from rag_benchmark.shared.records import QARecord
from rag_benchmark.techniques import TECHNIQUES, get_technique

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rankify_worker" / "src"))
from rag_benchmark_rankify.worker import (  # noqa: E402
    PRESETS,
    _capabilities,
    _configure_java_home,
    _custom_prompt,
    _generation_kwargs,
    _generator,
    _ensure_rankify_vllm_import,
    _reranker,
    _retrievers,
)
from rag_benchmark.techniques.rankify.config import RANKIFY_PRESETS, _worker_common
from rag_benchmark.techniques.rankify.pipeline import _convert_worker_rows, _question_payload

from rag_benchmark_rankify.config import PRESETS, _configure_java_home
from rag_benchmark_rankify.generation import _custom_prompt, _generation_kwargs, _generator
from rag_benchmark_rankify.retrieval import _reranker, _retrievers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rankify_worker" / "src"))
from rag_benchmark_rankify.worker import (  # noqa: E402
    PRESETS,
    _capabilities,
    _configure_java_home,
    _custom_prompt,
    _generation_kwargs,
    _generator,
    _ensure_rankify_vllm_import,
    _reranker,
    _retrievers,
)



def _fake_rankify(monkeypatch):
    package = ModuleType("rankify")
    package.__path__ = []
    generator_package = ModuleType("rankify.generator")
    generator_package.__path__ = []
    generator_module = ModuleType("rankify.generator.generator")
    models_package = ModuleType("rankify.models")
    models_package.__path__ = []
    transformer_module = ModuleType("rankify.models.transformer_ranker")
    retrievers_package = ModuleType("rankify.retrievers")
    retrievers_package.__path__ = []
    bge_module = ModuleType("rankify.retrievers.bge_retriever")
    bm25_module = ModuleType("rankify.retrievers.bm25_retriever")
    prompt_module = ModuleType("rankify.generator.prompt_template")
    vllm_module = ModuleType("vllm")

    class Generator:
        last_kwargs = None

        def __init__(self, **kwargs):
            type(self).last_kwargs = kwargs

    class TransformerRanker:
        last_kwargs = None

        def __init__(self, **kwargs):
            type(self).last_kwargs = kwargs

    class BGERetriever:
        last_kwargs = None

        def __init__(self, **kwargs):
            type(self).last_kwargs = kwargs

    class BM25Retriever:
        last_kwargs = None

        def __init__(self, **kwargs):
            type(self).last_kwargs = kwargs

    generator_module.Generator = Generator
    transformer_module.TransformerRanker = TransformerRanker
    bge_module.BGERetriever = BGERetriever
    bm25_module.BM25Retriever = BM25Retriever
    prompt_module.DEFAULT_PROMPTS = {"in-context-ralm": "native RALM prompt"}
    prompt_module.PromptTemplate = SimpleNamespace(IN_CONTEXT_RALM="in-context-ralm")
    for name, module in {
        "rankify": package,
        "rankify.generator": generator_package,
        "rankify.generator.generator": generator_module,
        "rankify.generator.prompt_template": prompt_module,
        "rankify.models": models_package,
        "rankify.models.transformer_ranker": transformer_module,
        "rankify.retrievers": retrievers_package,
        "rankify.retrievers.bge_retriever": bge_module,
        "rankify.retrievers.bm25_retriever": bm25_module,
        "vllm": vllm_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(__version__="1.26.4"))
    monkeypatch.setitem(sys.modules, "pyarrow", SimpleNamespace(__version__="17.0.0"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="2.5.1",
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda index: "Mock GPU"),
    ))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.1.4")
    return Generator, TransformerRanker, BM25Retriever, BGERetriever


def test_rrag_is_absent_from_registry_and_cli():
    assert tuple(TECHNIQUES) == ("dillema", "qwen3_embedding", "qwen3_embedding_rerank")
    with pytest.raises(ValueError, match="Unsupported"):
        get_technique("rrag")
    parser = build_parser()
    help_text = parser.format_help()
    assert "train" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--technique", "rrag"])


def test_benchmark_parser_accepts_detach_aliases():
    parser = build_parser()
    arguments = [
        "benchmark", "--config", "experiment.yaml", "--stage", "all",
        "--output", "artifacts/benchmarks/run",
    ]
    assert parser.parse_args(arguments).detach is False
    assert parser.parse_args([*arguments, "-d"]).detach is True
    assert parser.parse_args([*arguments, "--detach"]).detach is True


def test_detached_benchmark_launches_same_cli_and_tracks_pid(tmp_path, monkeypatch):
    config = tmp_path / "experiment.yaml"
    output = tmp_path / "run"
    output.mkdir()
    (output / "benchmark.pid").write_text("999\n", encoding="utf-8")
    monkeypatch.setattr("rag_benchmark.cli.platform.system", lambda: "Linux")
    monkeypatch.setattr(ExperimentSpec, "load", lambda path: SimpleNamespace())
    monkeypatch.setattr("rag_benchmark.cli._pid_is_running", lambda pid: False)
    observed = {}

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr("rag_benchmark.cli.subprocess.Popen", fake_popen)
    result = _launch_detached_benchmark(config, "all", output)

    assert result == {
        "status": "started",
        "pid": 4321,
        "log": str((output / "benchmark.log").resolve()),
        "pid_file": str((output / "benchmark.pid").resolve()),
        "output": str(output.resolve()),
    }
    assert observed["command"] == [
        sys.executable, "-u", "-m", "rag_benchmark", "benchmark",
        "--config", str(config.resolve()), "--stage", "all",
        "--output", str(output.resolve()),
    ]
    assert observed["kwargs"]["start_new_session"] is True
    assert observed["kwargs"]["close_fds"] is True
    assert observed["kwargs"]["stdin"] == subprocess.DEVNULL
    assert observed["kwargs"]["stderr"] == subprocess.STDOUT
    assert observed["kwargs"]["cwd"] == str(Path.cwd())
    assert observed["kwargs"]["env"]["PYTHONUNBUFFERED"] == "1"
    assert observed["kwargs"]["stdout"].closed
    assert (output / "benchmark.pid").read_text(encoding="utf-8") == "4321\n"


def test_detached_benchmark_rejects_live_pid(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    (output / "benchmark.pid").write_text("123\n", encoding="utf-8")
    monkeypatch.setattr("rag_benchmark.cli.platform.system", lambda: "Linux")
    monkeypatch.setattr(ExperimentSpec, "load", lambda path: SimpleNamespace())
    monkeypatch.setattr("rag_benchmark.cli._pid_is_running", lambda pid: True)

    with pytest.raises(RuntimeError, match="already running with PID 123"):
        _launch_detached_benchmark(tmp_path / "experiment.yaml", "all", output)


def test_detached_benchmark_preflights_huggingface_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "dataset"
    source.mkdir()
    monkeypatch.setenv("HF_MODEL_NAME", str((tmp_path / "missing-snapshot").resolve()))
    monkeypatch.setattr("rag_benchmark.cli.platform.system", lambda: "Linux")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join([
            "dataset: its", "source: dataset", "index_root: indexes",
            "rankify_python: python", "generator:", "  backend: huggingface",
            "  model_env: HF_MODEL_NAME", "  temperature: 0", "models:",
            "  dense_multilingual: configurable/dense",
            "  reranker_multilingual: configurable/reranker",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "rag_benchmark.cli.subprocess.Popen",
        lambda *args, **kwargs: pytest.fail("Popen must not run when preflight fails"),
    )

    with pytest.raises(FileNotFoundError, match="snapshot config not found"):
        _launch_detached_benchmark(config, "all", tmp_path / "run")


def test_setup_scripts_use_uv_only():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        (root / "scripts" / "setup_ubuntu.sh").read_text(encoding="utf-8"),
        (root / "scripts" / "setup_rankify_ubuntu.sh").read_text(encoding="utf-8"),
        (root / "scripts" / "setup_rankify_gpu_ubuntu.sh").read_text(encoding="utf-8"),
        (root / "scripts" / "setup_vllm_gpu_ubuntu.sh").read_text(encoding="utf-8"),
    ]
    assert all("uv sync" in script for script in scripts)
    assert all("pip install" not in script for script in scripts)
    assert all("python3 -m venv" not in script for script in scripts)


def test_rankify_gpu_environment_targets_blackwell():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads(
        (root / "rankify_worker_gpu" / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = project["project"]["dependencies"]

    assert "torch==2.13.0" in dependencies
    assert "rankify[retriever,rag]==0.1.4" in dependencies
    assert all("reranking" not in dependency for dependency in dependencies)
    assert project["tool"]["uv"]["index"][0]["url"].endswith("/cu132")
    assert project["tool"]["uv"]["sources"]["dillema-rankify-worker"]["path"].endswith(
        "src/rag_benchmark/techniques/rankify/worker"
    )
    rankify_lock = tomllib.loads(
        (root / "rankify_worker_gpu" / "uv.lock").read_text(encoding="utf-8")
    )
    assert all(package["name"] != "vllm" for package in rankify_lock["package"])

    for config in (root / "configs" / "experiments").glob("rankify-*.yaml"):
        assert "../../.venv-rankify-gpu/bin/python" in config.read_text(encoding="utf-8")


def test_main_gpu_environment_targets_blackwell():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    setup = (root / "scripts" / "setup_ubuntu.sh").read_text(encoding="utf-8")

    assert "torch==2.13.0" in project["project"]["optional-dependencies"]["hf"]
    assert "bitsandbytes==0.50.2" in project["project"]["optional-dependencies"]["quantized"]
    assert project["tool"]["uv"]["index"][0]["url"].endswith("/cu132")
    assert "--locked" in setup
    assert 'startswith("13.2")' in setup
    assert "torch.cuda.get_device_capability(0)" in setup


def test_vllm_gpu_environment_is_isolated_and_pinned():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads(
        (root / "vllm_gpu" / "pyproject.toml").read_text(encoding="utf-8")
    )
    setup = (root / "scripts" / "setup_vllm_gpu_ubuntu.sh").read_text(encoding="utf-8")
    runner = (root / "scripts" / "run_vllm_gpu_ubuntu.sh").read_text(encoding="utf-8")
    lock = tomllib.loads((root / "vllm_gpu" / "uv.lock").read_text(encoding="utf-8"))
    locked = {package["name"]: package["version"] for package in lock["package"]}

    assert project["project"]["dependencies"] == ["vllm==0.32.0"]
    assert locked["vllm"] == "0.32.0"
    assert locked["torch"] == "2.13.0"
    assert ".venv-vllm-gpu" in setup
    assert "torch.cuda.get_device_capability(0)" in setup
    assert 'startswith("13.0")' in setup
    assert "from vllm import LLM" in setup
    assert "LLM_MODEL" in runner
    assert "LLM_API_KEY" in runner
    assert "VLLM_GPU_MEMORY_UTILIZATION:-0.65" in runner


def test_rankify_vllm_import_shim_fails_clearly_when_used(monkeypatch):
    monkeypatch.delitem(sys.modules, "vllm", raising=False)

    def missing_vllm(name):
        assert name == "vllm"
        raise ModuleNotFoundError("No module named 'vllm'", name="vllm")

    monkeypatch.setattr("rag_benchmark_rankify.worker.importlib.import_module", missing_vllm)

    assert _ensure_rankify_vllm_import() is True
    with pytest.raises(RuntimeError, match="openai-compatible"):
        sys.modules["vllm"].LLM(model="unused")


def test_rankify_worker_configures_java_home_from_javac(tmp_path, monkeypatch):
    javac = tmp_path / "jdk-21" / "bin" / "javac"
    javac.parent.mkdir(parents=True)
    javac.write_text("", encoding="utf-8")
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr("rag_benchmark_rankify.config.shutil.which", lambda name: str(javac))

    assert _configure_java_home() == str(javac.parent.parent.resolve())
    assert os.environ["JAVA_HOME"] == str(javac.parent.parent.resolve())


def test_rankify_worker_requires_full_jdk(monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr("rag_benchmark_rankify.config.shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="openjdk-21-jdk-headless"):
        _configure_java_home()


def test_generation_and_experiment_models_are_configuration_driven(tmp_path, monkeypatch):
    source = tmp_path / "dataset"
    source.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_MODEL_NAME", str(snapshot.resolve()))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join([
            "dataset: its",
            "source: dataset",
            "index_root: indexes",
            "rankify_python: python",
            "evaluation:",
            "  split: all",
            "generator:",
            "  backend: huggingface",
            "  model_env: HF_MODEL_NAME",
            "  temperature: 0",
            "  max_tokens: 64",
            "models:",
            "  dense_multilingual: configurable/dense",
            "  reranker_multilingual: configurable/reranker",
            "retrieval:",
            "  candidates: 40",
            "  top_k: 10",
            "  rrf_k: 60",
        ]),
        encoding="utf-8",
    )
    spec = ExperimentSpec.load(config)
    assert spec.generator.model == str(snapshot.resolve())
    assert spec.generator.backend == "huggingface"
    assert spec.dense_model == "configurable/dense"
    assert spec.evaluation_split == "all"
    assert spec.public_dict()["evaluation"] == {"split": "all"}
    assert spec.presets == RANKIFY_PRESETS
    assert _worker_common(spec)["generator"]["model"] == str(snapshot.resolve())
    first = spec.generator.fingerprint()
    second = GenerationSpec.from_dict({
        "backend": "huggingface", "model": "small/model-b", "temperature": 0,
    }).fingerprint()
    assert first != second


def test_experiment_rejects_unknown_evaluation_split(tmp_path, monkeypatch):
    source = tmp_path / "dataset"
    source.mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_MODEL_NAME", str(snapshot.resolve()))
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join([
            "dataset: its", "source: dataset", "index_root: indexes",
            "rankify_python: python", "evaluation:", "  split: invalid",
            "generator:", "  backend: huggingface", "  model_env: HF_MODEL_NAME",
            "  temperature: 0", "models:",
            "  dense_multilingual: configurable/dense",
            "  reranker_multilingual: configurable/reranker",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation.split must be test or all"):
        ExperimentSpec.load(config)


def test_huggingface_experiment_rejects_missing_local_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "dataset"
    source.mkdir()
    missing = tmp_path / "missing-snapshot"
    monkeypatch.setenv("HF_MODEL_NAME", str(missing.resolve()))
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "\n".join([
            "dataset: its", "source: dataset", "index_root: indexes",
            "rankify_python: python", "generator:", "  backend: huggingface",
            "  model_env: HF_MODEL_NAME", "  temperature: 0", "models:",
            "  dense_multilingual: configurable/dense",
            "  reranker_multilingual: configurable/reranker",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="snapshot config not found"):
        ExperimentSpec.load(config)


def test_worker_exposes_the_controlled_seven_presets():
    assert tuple(PRESETS) == RANKIFY_PRESETS


def test_rankify_generation_is_greedy_and_bounded():
    api = {"generator": {"backend": "openai-compatible", "max_tokens": 123}}
    local = {"generator": {"backend": "huggingface", "max_tokens": 123}}
    assert _generation_kwargs(api) == {"temperature": 0, "max_tokens": 123}
    assert _generation_kwargs(local) == {"do_sample": False, "max_new_tokens": 123}


def test_rankify_capability_probe_and_openai_compatible_configuration(monkeypatch):
    generator_class, reranker_class, bm25_class, bge_class = _fake_rankify(monkeypatch)
    capabilities = _capabilities()
    assert capabilities["rankify_version"] == "0.1.4"
    assert capabilities["reranker_import"] == "TransformerRanker"
    assert capabilities["numpy_version"] == "1.26.4"
    assert capabilities["pyarrow_version"] == "17.0.0"
    assert capabilities["cuda_available"] is True
    assert capabilities["rankify_vllm_import"] == "installed"
    assert "rankify.models.reranking" not in sys.modules
    assert "rankify.retrievers.retriever" not in sys.modules
    assert capabilities["retriever_imports"] == ["BM25Retriever", "BGERetriever"]
    monkeypatch.setenv("LLM_API_KEY", "runtime-only-secret")
    _generator({"generator": {
        "backend": "openai-compatible",
        "model": "openrouter/free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "LLM_API_KEY",
        "max_tokens": 768,
    }}, "basic-rag", "cuda")
    assert generator_class.last_kwargs == {
        "method": "basic-rag",
        "model_name": "openrouter/free",
        "backend": "openai",
        "api_keys": ["runtime-only-secret"],
        "base_url": "https://openrouter.ai/api/v1",
    }
    _generator({"generator": {
        "backend": "openai-compatible", "model": "openrouter/free",
        "base_url": "https://openrouter.ai/api/v1", "api_key_env": "LLM_API_KEY",
    }}, "in-context-ralm", "cuda")
    assert generator_class.last_kwargs["method"] == "basic-rag"
    assert _custom_prompt("in-context-ralm") == "native RALM prompt"
    _reranker({"reranker_model": "BAAI/bge-reranker-v2-m3"}, True, "cuda")
    assert reranker_class.last_kwargs == {
        "method": "transformer_ranker",
        "model_name": "BAAI/bge-reranker-v2-m3",
        "device": "cuda",
    }
    retrievers = _retrievers(
        {"candidates": 40, "dense_model": "BAAI/bge-m3"},
        {"index_dir": "/tmp/index"}, "hybrid", "cuda",
    )
    assert set(retrievers) == {"bm25", "bge"}
    assert bm25_class.last_kwargs == {
        "n_docs": 40,
        "index_type": "custom",
        "index_folder": str(Path("/tmp/index/bm25")),
    }
    assert bge_class.last_kwargs == {
        "n_docs": 40, "index_folder": str(Path("/tmp/index/bge/bge_index_custom")),
        "model": "BAAI/bge-m3", "device": "cuda",
    }


def test_rankify_question_payload_never_contains_references():
    question = QARecord(
        "q1", "its", "id", "Pertanyaan?", ["jawaban rahasia"], "test",
        source_ids=["doc-1"], metadata={"category": "akademik"},
    )
    payload = _question_payload(question)
    assert "references" not in payload
    assert "jawaban rahasia" not in json.dumps(payload, ensure_ascii=False)


def test_api_secret_is_not_serialized(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "do-not-persist")
    spec = GenerationSpec.from_dict({
        "backend": "openai-compatible", "model": "provider/model",
        "base_url": "https://example.test/v1", "api_key_env": "LLM_API_KEY",
        "temperature": 0,
    })
    assert "do-not-persist" not in json.dumps(spec.public_dict())


def test_worker_rows_join_references_only_after_generation():
    question = QARecord(
        "q1", "its", "id", "Pertanyaan?", ["jawaban benar"], "test",
        source_ids=["doc-1"], metadata={"relevance_label": "weak/category-based"},
    )
    spec = SimpleNamespace(
        generator=GenerationSpec.from_dict({
            "backend": "huggingface", "model": "small/model", "temperature": 0,
        }),
    )
    rows = _convert_worker_rows([{
        "id": "q1", "prediction": "jawaban benar", "status": "ok", "latency_ms": 1,
        "context_length": 10, "diagnostics": {},
        "evidence": [{
            "source_id": "doc-1", "chunk_id": "chunk-1", "text": "context",
            "score": 0.9, "metadata": {},
        }],
    }], [question], "rankify_bm25_basic", spec, "corpus")
    assert rows[0]["references"] == ["jawaban benar"]
    assert rows[0]["diagnostics"]["relevant_ranks"] == [1]


def test_comparison_rejects_decoding_or_corpus_mismatch(tmp_path):
    common = {
        "id": "q1", "dataset": "its", "language": "id", "split": "test",
        "question": "q", "references": ["a b c d"], "prediction": "a b c d",
        "status": "ok", "available": True, "model_fingerprint": "model",
        "gold_source_ids": [], "diagnostics": {}, "latency_ms": 1,
    }
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(baseline, [{**common, "technique": "dillema", "corpus_fingerprint": "one", "decoding_config": {"max_tokens": 10}}])
    write_jsonl(candidate, [{**common, "technique": "rankify_bm25_basic", "corpus_fingerprint": "two", "decoding_config": {"max_tokens": 10}}])
    with pytest.raises(ValueError, match="fingerprints"):
        run_compare_techniques(baseline, candidate, tmp_path / "out")
