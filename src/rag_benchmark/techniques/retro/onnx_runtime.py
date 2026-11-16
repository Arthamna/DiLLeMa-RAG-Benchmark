from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import time
import gc
from pathlib import Path
from typing import Any

from ..piperag.schedule import StaleAsyncSchedule
from .client import RetrievalServiceClient
from .config import ServiceConfig
from .prompt import build_qa_prompt_ids, extract_answer
from ...shared.config import collection_name
from ...shared.device import cuda_architecture, cuda_architecture_supported, release_cuda_memory


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_digest() -> str:
    from ..._vendor.piperag import modeling_retro_inference

    return sha256_file(Path(modeling_retro_inference.__file__).resolve())


def _runtime_versions() -> dict[str, str]:
    packages = ("torch", "onnx", "onnxruntime-gpu")
    return {name: importlib.metadata.version(name) for name in packages}


def _model_fingerprint(checkpoint: Path, model_config: Path) -> str:
    payload = ":".join((
        sha256_file(checkpoint), sha256_file(model_config), _runtime_digest(),
        json.dumps(_runtime_versions(), sort_keys=True, separators=(",", ":")),
    ))
    return hashlib.sha256(payload.encode()).hexdigest()


def uses_stale_schedule(mode: str, stale_steps: int) -> bool:
    return mode == "piperag" and int(stale_steps) > 0


def service_retrieval_endpoint(request: dict[str, Any]) -> str:
    """Prefer the Compose-internal retrieval address when one is configured."""

    return str(os.getenv("RETRO_RETRIEVAL_ENDPOINT") or request["retrieval_endpoint"])


def cuda_provider_options() -> dict[str, str]:
    strategy = os.getenv("RETRO_ORT_ARENA_EXTEND_STRATEGY", "kSameAsRequested").strip()
    if strategy not in {"kSameAsRequested", "kNextPowerOfTwo"}:
        raise ValueError(
            "RETRO_ORT_ARENA_EXTEND_STRATEGY must be kSameAsRequested or kNextPowerOfTwo"
        )
    return {"arena_extend_strategy": strategy}


def clone_past_key_values(values):
    """Isolate mutable decoder cache inputs for PyTorch/ONNX parity checks."""

    return [value.clone() if hasattr(value, "clone") else value.copy() for value in values]


def parity_report(
    expected_encoder, actual_encoder, expected_logits, actual_logits,
    encoder_tolerance: float, decoder_tolerance: float | None = None,
) -> dict[str, Any]:
    import numpy as np

    decoder_tolerance = encoder_tolerance if decoder_tolerance is None else decoder_tolerance
    encoder_error = max(
        float(np.max(np.abs(actual_encoder[index] - expected_encoder[index]))) for index in range(2)
    )
    decoder_error = float(np.max(np.abs(actual_logits - expected_logits)))
    token_equal = int(np.argmax(actual_logits[0, -1])) == int(np.argmax(expected_logits[0, -1]))
    if not token_equal or decoder_error > decoder_tolerance or encoder_error > encoder_tolerance:
        raise RuntimeError(
            "PyTorch/ONNX parity failed: "
            f"token_equal={token_equal}, "
            f"encoder_max_abs={encoder_error} (atol={encoder_tolerance}), "
            f"decoder_max_abs={decoder_error} (atol={decoder_tolerance})"
        )
    return {
        "token_equal": token_equal, "encoder_max_abs": encoder_error,
        "decoder_max_abs": decoder_error,
        "encoder_atol": encoder_tolerance, "decoder_atol": decoder_tolerance,
    }


class OnnxRetroRuntime:
    """Shared RETRO/PipeRAG runtime derived from PipeRAG's ONNX client.

    PyTorch is used only to load the checkpoint, export a fingerprinted ONNX
    pair, and bootstrap cache tensors for each prompt. Token decoding and
    neighbour encoding run in ONNX Runtime with CUDA.
    """

    def __init__(self):
        self.protocol_version = 1
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.encoder_session = None
        self.decoder_session = None
        self.retrieval: RetrievalServiceClient | None = None
        self.configured: dict[str, Any] = {}
        self.model_fingerprint = ""
        self.asset_manifest: dict[str, Any] = {}
        self.bootstrap_device = ""

    @staticmethod
    def environment() -> dict[str, Any]:
        result: dict[str, Any] = {"cuda": False, "onnx_providers": [], "device": None}
        try:
            import torch

            result["cuda"] = bool(torch.cuda.is_available())
            result["device"] = torch.cuda.get_device_name(0) if result["cuda"] else None
            result["torch"] = torch.__version__
            result["torch_cuda"] = torch.version.cuda
            if result["cuda"]:
                capability = tuple(torch.cuda.get_device_capability(0))
                architectures = list(torch.cuda.get_arch_list())
                result["cuda_capability"] = list(capability)
                result["cuda_architecture"] = cuda_architecture(capability)
                result["torch_architectures"] = architectures
                result["architecture_supported"] = cuda_architecture_supported(capability, architectures)
        except Exception as exc:
            result["torch_error"] = str(exc)
        try:
            import onnxruntime as ort

            result["onnx_providers"] = list(ort.get_available_providers())
            result["onnxruntime"] = ort.__version__
        except Exception as exc:
            result["onnx_error"] = str(exc)
        return result

    def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        required = ("checkpoint", "model_config", "onnx_encoder", "onnx_decoder", "retrieval_endpoint")
        missing = [name for name in required if not str(request.get(name) or "").strip()]
        if missing:
            raise ValueError(f"Missing inference configuration: {', '.join(missing)}")
        paths = {name: Path(str(request[name])).expanduser().resolve() for name in required[:4]}
        for name in ("checkpoint", "model_config"):
            if not paths[name].is_file():
                raise FileNotFoundError(f"{name} not found inside inference container: {paths[name]}")
        environment = self.environment()
        require_cuda = os.getenv("RETRO_REQUIRE_CUDA", "1") != "0"
        if require_cuda and not environment.get("cuda"):
            raise RuntimeError("RETRO inference requires an NVIDIA CUDA device")
        if require_cuda and environment.get("torch_error"):
            raise RuntimeError(f"PyTorch CUDA probe failed: {environment['torch_error']}")
        if require_cuda and "CUDAExecutionProvider" not in environment.get("onnx_providers", []):
            raise RuntimeError("onnxruntime does not expose CUDAExecutionProvider")
        if require_cuda and environment.get("architecture_supported") is False:
            raise RuntimeError(
                f"PyTorch wheel does not support {environment.get('cuda_architecture')}; "
                f"available architectures: {environment.get('torch_architectures')}"
            )
        bootstrap_device = os.getenv("RETRO_BOOTSTRAP_DEVICE", "cpu").strip().casefold()
        if bootstrap_device not in {"cpu", "cuda"}:
            raise ValueError("RETRO_BOOTSTRAP_DEVICE must be cpu or cuda")
        if bootstrap_device == "cuda" and not environment.get("cuda"):
            raise RuntimeError("RETRO_BOOTSTRAP_DEVICE=cuda requires an NVIDIA CUDA device")

        next_fingerprint = _model_fingerprint(paths["checkpoint"], paths["model_config"])
        stable_fields = ("checkpoint", "model_config", "onnx_encoder", "onnx_decoder", "tokenizer", "retro")
        reuse_loaded_model = bool(
            self.model is not None
            and self.encoder_session is not None
            and self.decoder_session is not None
            and self.model_fingerprint == next_fingerprint
            and self.bootstrap_device == bootstrap_device
            and all(self.configured.get(name) == request.get(name) for name in stable_fields)
        )
        if not reuse_loaded_model:
            self.unload()
        self.configured = dict(request)
        self.model_fingerprint = next_fingerprint
        self.bootstrap_device = bootstrap_device
        if not reuse_loaded_model:
            self._load_torch_model(paths["checkpoint"], paths["model_config"])
            self.model = self.model.to(bootstrap_device)
            if hasattr(self.model, "model_device"):
                self.model.model_device = bootstrap_device
            if bootstrap_device == "cpu":
                release_cuda_memory("RETRO PyTorch bootstrap model CPU offload")
            self.asset_manifest = self._ensure_onnx(paths["onnx_encoder"], paths["onnx_decoder"])
            self._load_onnx(paths["onnx_encoder"], paths["onnx_decoder"], require_cuda)
        service_config = ServiceConfig(
            retrieval_endpoint=service_retrieval_endpoint(request),
            inference_endpoint="unused:0",
            timeout_seconds=float(request.get("service_timeout_seconds", 120.0)),
            protocol_version=int(request.get("protocol_version", self.protocol_version)),
            max_message_mb=int(request.get("max_message_mb", 64)),
        )
        if self.retrieval is not None:
            self.retrieval.close()
        self.retrieval = RetrievalServiceClient(service_config)
        return {
            "model_fingerprint": self.model_fingerprint,
            "checkpoint_sha256": sha256_file(paths["checkpoint"]),
            "onnx": self.asset_manifest,
            "environment": environment,
            "model_reused": reuse_loaded_model,
            "bootstrap_device": self.bootstrap_device,
        }

    def _load_torch_model(self, checkpoint: Path, config_path: Path) -> None:
        import torch
        from transformers import AutoTokenizer

        from ..._vendor.piperag import RetroConfig, RetroModelLMHeadInference
        from ..._vendor.piperag.retrieval import DummyRetriever

        value = json.loads(config_path.read_text(encoding="utf-8"))
        self.model_config = RetroConfig(**value)
        num_neighbours = int((self.configured.get("retro") or {}).get("num_neighbours", 2))
        continuation = int((self.configured.get("retro") or {}).get("continuation_chunks", 1))
        retriever = DummyRetriever(num_neighbours, self.model_config.chunk_size * (1 + continuation))
        # Keep checkpoint loading on CPU.  The ONNX CUDA sessions own the GPU;
        # retaining a second full PyTorch copy there leaves too little room for
        # the growing decoder KV cache during 1,024-token paper windows.
        model = RetroModelLMHeadInference(self.model_config, retriever=retriever, device="cpu")
        checkpoint_value = torch.load(str(checkpoint), map_location="cpu")
        state = checkpoint_value.get("state_dict", checkpoint_value)
        if not isinstance(state, dict):
            raise ValueError("Checkpoint does not contain a state_dict")
        normalized = {}
        for key, tensor in state.items():
            name = str(key)
            for prefix in ("model.", "module."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            normalized[name] = tensor
        incompatible = model.load_state_dict(normalized, strict=False)
        unexpected = [name for name in incompatible.unexpected_keys if not name.endswith("num_batches_tracked")]
        if incompatible.missing_keys or unexpected:
            raise ValueError(
                "Checkpoint is incompatible with the pinned PipeRAG inference model: "
                f"missing={incompatible.missing_keys[:8]}, unexpected={unexpected[:8]}"
            )
        self.model = model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.configured.get("tokenizer") or "t5-base"), use_fast=True,
        )

    def _torch_states(self, input_ids, neighbour_ids):
        import torch

        model = self.model
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        neighbour_ids = neighbour_ids.to(device)
        hidden = torch.zeros(
            (neighbour_ids.shape[0], 1, neighbour_ids.shape[2], neighbour_ids.shape[3], model.config.enc_hidden_dim),
            dtype=torch.float32, device=device,
        )
        mask = torch.zeros(neighbour_ids.shape, dtype=torch.float32, device=device)
        with torch.inference_mode():
            _, _, ca_hidden, ca_mask = model(input_ids, hidden, mask)
            neighbour_hidden, neighbour_mask = model.base.encoder(neighbour_ids, ca_hidden, ca_mask)
            seed = input_ids[:, :-1] if input_ids.shape[1] > 1 else input_ids
            _, past, ca_hidden, ca_mask = model(
                seed, neighbour_hidden, neighbour_mask,
                past_key_value=None, past_ca_hidden_states=None,
            )
        return neighbour_hidden, neighbour_mask, list(past), ca_hidden, ca_mask

    @staticmethod
    def _to_numpy(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else value

    def _ensure_onnx(self, encoder_path: Path, decoder_path: Path) -> dict[str, Any]:
        manifest_path = decoder_path.with_suffix(decoder_path.suffix + ".manifest.json")
        if encoder_path.is_file() and decoder_path.is_file() and manifest_path.is_file():
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            if current.get("model_fingerprint") == self.model_fingerprint:
                if current.get("encoder_sha256") == sha256_file(encoder_path) and current.get("decoder_sha256") == sha256_file(decoder_path):
                    return {**current, "cache": "used"}
        encoder_path.parent.mkdir(parents=True, exist_ok=True)
        decoder_path.parent.mkdir(parents=True, exist_ok=True)
        self._export_onnx(encoder_path, decoder_path)
        parity = self._validate_onnx(encoder_path, decoder_path)
        manifest = {
            "schema_version": 1,
            "model_fingerprint": self.model_fingerprint,
            "encoder_sha256": sha256_file(encoder_path),
            "decoder_sha256": sha256_file(decoder_path),
            "runtime_sha256": _runtime_digest(),
            "runtime_versions": _runtime_versions(),
            "parity": parity,
            "cache": "rebuilt",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest

    def _export_onnx(self, encoder_path: Path, decoder_path: Path) -> None:
        import torch

        retro = self.configured.get("retro") or {}
        chunk_size = int(retro.get("chunk_size", 64))
        neighbours = int(retro.get("num_neighbours", 2))
        neighbour_len = chunk_size * (1 + int(retro.get("continuation_chunks", 1)))
        device = next(self.model.parameters()).device
        input_ids = torch.zeros((1, chunk_size), dtype=torch.int64, device=device)
        neighbour_ids = torch.zeros((1, 1, neighbours, neighbour_len), dtype=torch.int64, device=device)
        neighbour_hidden, neighbour_mask, past, ca_hidden, ca_mask = self._torch_states(input_ids, neighbour_ids)
        torch.onnx.export(
            self.model.base.encoder,
            (neighbour_ids, ca_hidden, ca_mask),
            str(encoder_path),
            export_params=True,
            dynamo=False,
            opset_version=17,
            input_names=["neighbour_ids", "past_ca_hidden_states", "past_ca_attention_mask"],
            output_names=["neighbour_hidden_states", "neighbour_attention_mask"],
            dynamic_axes={
                "neighbour_ids": {2: "neighbours", 3: "neighbour_tokens"},
                "past_ca_hidden_states": {2: "ca_tokens"},
                "past_ca_attention_mask": {2: "ca_tokens"},
            },
        )
        input_names = ["input_ids", "neighbour_hidden_states", "neighbour_attention_mask"]
        input_names += [f"past_key_value.{index}" for index in range(len(past))]
        input_names += ["past_ca_hidden_states"]
        output_names = ["logits"] + [f"out_key_value.{index}" for index in range(len(past))]
        output_names += ["out_ca_hidden_states", "out_ca_attention_mask"]
        dynamic_axes = {
            "input_ids": {1: "tokens"},
            "neighbour_hidden_states": {2: "neighbours", 3: "neighbour_tokens"},
            "neighbour_attention_mask": {2: "neighbours", 3: "neighbour_tokens"},
            "past_ca_hidden_states": {2: "ca_tokens"},
            "logits": {1: "tokens"},
        }
        for index in range(len(past)):
            dynamic_axes[f"past_key_value.{index}"] = {2: "past_tokens"}
            dynamic_axes[f"out_key_value.{index}"] = {2: "out_tokens"}
        torch.onnx.export(
            self.model,
            (input_ids, neighbour_hidden, neighbour_mask, past, ca_hidden),
            str(decoder_path),
            export_params=True,
            dynamo=False,
            opset_version=17,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    @staticmethod
    def _sessions(encoder_path: Path, decoder_path: Path, require_cuda: bool):
        # Importing PyTorch first preloads the CUDA/cuDNN shared libraries from
        # its wheel.  Without this, a host install can fail while importing ORT
        # with `libcudart.so.13: cannot open shared object file`.
        import torch  # noqa: F401
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", cuda_provider_options()))
        providers.append("CPUExecutionProvider")
        encoder = ort.InferenceSession(str(encoder_path), options, providers=providers)
        decoder = ort.InferenceSession(str(decoder_path), options, providers=providers)
        if require_cuda and (encoder.get_providers()[0] != "CUDAExecutionProvider" or decoder.get_providers()[0] != "CUDAExecutionProvider"):
            raise RuntimeError("ONNX models did not initialize on CUDAExecutionProvider")
        return encoder, decoder

    def _validate_onnx(self, encoder_path: Path, decoder_path: Path) -> dict[str, Any]:
        import numpy as np
        import torch

        require_cuda = os.getenv("RETRO_REQUIRE_CUDA", "1") != "0"
        encoder, decoder = self._sessions(encoder_path, decoder_path, require_cuda)
        retro = self.configured.get("retro") or {}
        chunk_size = int(retro.get("chunk_size", 64))
        neighbours = int(retro.get("num_neighbours", 2))
        neighbour_len = chunk_size * (1 + int(retro.get("continuation_chunks", 1)))
        device = next(self.model.parameters()).device
        torch.manual_seed(42)
        input_ids = torch.randint(0, min(100, self.model.config.num_embeddings), (1, chunk_size), device=device)
        neighbour_ids = torch.randint(0, min(100, self.model.config.num_embeddings), (1, 1, neighbours, neighbour_len), device=device)
        neighbour_hidden, neighbour_mask, past, ca_hidden, ca_mask = self._torch_states(input_ids, neighbour_ids)
        encoded = encoder.run(None, {
            "neighbour_ids": self._to_numpy(neighbour_ids),
            "past_ca_hidden_states": self._to_numpy(ca_hidden),
            "past_ca_attention_mask": self._to_numpy(ca_mask),
        })
        decoder_inputs = {
            "input_ids": self._to_numpy(input_ids),
            "neighbour_hidden_states": self._to_numpy(neighbour_hidden),
            "neighbour_attention_mask": self._to_numpy(neighbour_mask),
            "past_ca_hidden_states": self._to_numpy(ca_hidden),
        }
        decoder_inputs.update({f"past_key_value.{i}": self._to_numpy(value) for i, value in enumerate(past)})
        with torch.inference_mode():
            expected_encoder = self.model.base.encoder(neighbour_ids, ca_hidden, ca_mask)
            expected_decoder = self.model(
                input_ids, neighbour_hidden, neighbour_mask,
                clone_past_key_values(past), ca_hidden,
            )[0]
        decoded = decoder.run(None, decoder_inputs)
        expected_logits = self._to_numpy(expected_decoder)
        legacy_tolerance = os.getenv("RETRO_ONNX_ATOL")
        encoder_tolerance = float(legacy_tolerance or os.getenv("RETRO_ONNX_ENCODER_ATOL", "0.002"))
        decoder_tolerance = float(legacy_tolerance or os.getenv("RETRO_ONNX_DECODER_ATOL", "0.002"))
        return parity_report(
            [self._to_numpy(value) for value in expected_encoder], encoded,
            expected_logits, decoded[0], encoder_tolerance, decoder_tolerance,
        )

    def _load_onnx(self, encoder_path: Path, decoder_path: Path, require_cuda: bool) -> None:
        self.encoder_session, self.decoder_session = self._sessions(encoder_path, decoder_path, require_cuda)

    def _retrieve(self, index_id: str, token_ids: list[int], settings: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        import numpy as np

        query = self.tokenizer.decode(token_ids[-int(settings["chunk_size"]):], skip_special_tokens=True)
        started = time.perf_counter()
        response = self.retrieval.retrieve({
            "index_id": index_id,
            "queries": [query],
            "num_neighbours": int(settings["num_neighbours"]),
            "continuation_chunks": int(settings["continuation_chunks"]),
            "chunk_size": int(settings["chunk_size"]),
            "stale_steps": int(settings.get("stale_steps", 0)),
            "interval": int(settings["retrieval_interval"]),
            "seq_len": len(token_ids),
            "nprobe": int(settings["nprobe"]),
            "dynamic_nprobe": bool(settings.get("dynamic_nprobe", False)),
        })
        wait_ms = (time.perf_counter() - started) * 1000
        hits = list((response.get("results") or [{}])[0].get("hits") or [])
        neighbour_len = int(settings["chunk_size"]) * (1 + int(settings["continuation_chunks"]))
        values = np.zeros((1, 1, int(settings["num_neighbours"]), neighbour_len), dtype=np.int64)
        for index, hit in enumerate(hits[: int(settings["num_neighbours"])]):
            tokens = list(hit.get("token_ids") or [])[:neighbour_len]
            values[0, 0, index, :len(tokens)] = tokens
        return values, {"hits": hits, "wait_latency_ms": wait_ms, **dict(response.get("diagnostics") or {})}

    def _bootstrap(self, input_ids: list[int], neighbour_ids):
        import torch

        device = next(self.model.parameters()).device
        ids = torch.tensor([input_ids], dtype=torch.int64, device=device)
        neighbours = torch.tensor(neighbour_ids, dtype=torch.int64, device=device)
        neighbour_hidden, neighbour_mask, past, ca_hidden, ca_mask = self._torch_states(ids, neighbours)
        return (
            self._to_numpy(neighbour_hidden), self._to_numpy(neighbour_mask),
            [self._to_numpy(value) for value in past], self._to_numpy(ca_hidden), self._to_numpy(ca_mask),
        )

    def _decode_step(self, input_ids, neighbour_hidden, neighbour_mask, past, ca_hidden):
        inputs = {
            "input_ids": input_ids,
            "neighbour_hidden_states": neighbour_hidden,
            "neighbour_attention_mask": neighbour_mask,
            "past_ca_hidden_states": ca_hidden,
        }
        inputs.update({f"past_key_value.{i}": value for i, value in enumerate(past)})
        outputs = self.decoder_session.run(None, inputs)
        return outputs[0], list(outputs[1:-2]), outputs[-2], outputs[-1]

    def _encode(self, neighbour_ids, ca_hidden, ca_mask):
        outputs = self.encoder_session.run(None, {
            "neighbour_ids": neighbour_ids,
            "past_ca_hidden_states": ca_hidden,
            "past_ca_attention_mask": ca_mask,
        })
        return outputs[0], outputs[1]

    def _settings(self, request: dict[str, Any]) -> dict[str, Any]:
        retro = self.configured.get("retro") or {}
        mode = str(request.get("mode") or "retro")
        interval = int(request.get("retrieval_interval", retro.get("retrieval_interval", 64)))
        return {
            "mode": mode,
            "chunk_size": int(request.get("chunk_size", retro.get("chunk_size", 64))),
            "num_neighbours": int(request.get("num_neighbours", retro.get("num_neighbours", 2))),
            "continuation_chunks": int(request.get("continuation_chunks", retro.get("continuation_chunks", 1))),
            "retrieval_interval": interval,
            "stale_steps": int(request.get("stale_steps", interval if mode == "piperag" else 0)),
            "nprobe": int(request.get("nprobe", retro.get("nprobe", 16))),
            "dynamic_nprobe": bool(request.get("dynamic_nprobe", False)),
            "max_new_tokens": int(request.get("max_new_tokens", retro.get("max_new_tokens", 128))),
        }

    def _run_tokens(
        self, index_id: str, initial_ids: list[int], settings: dict[str, Any],
        *, continuation: list[int] | None = None,
    ) -> dict[str, Any]:
        import numpy as np

        if len(initial_ids) != int(settings["chunk_size"]):
            raise ValueError("RETRO input must be left-padded to exactly one 64-token chunk")
        retrieval_calls: list[dict[str, Any]] = []
        decoder_ms: list[float] = []
        encoder_ms: list[float] = []
        initial_neighbours, first = self._retrieve(index_id, initial_ids, settings)
        retrieval_calls.append(first)
        neighbour_hidden, neighbour_mask, past, ca_hidden, ca_mask = self._bootstrap(initial_ids, initial_neighbours)

        def retrieve(query: str, seq_len: int):
            del query
            values, diagnostics = self._retrieve(index_id, token_ids[:seq_len], settings)
            retrieval_calls.append(diagnostics)
            return values, diagnostics

        schedule = StaleAsyncSchedule(retrieve) if uses_stale_schedule(settings["mode"], settings["stale_steps"]) else None
        if schedule is not None:
            schedule.current = (initial_neighbours, first)
            schedule.calls = 1
        token_ids = list(initial_ids)
        nll = 0.0
        scored = 0
        target_values = list(continuation or [])
        steps = len(target_values) if continuation is not None else int(settings["max_new_tokens"])
        try:
            for step in range(steps):
                started = time.perf_counter()
                logits, past, ca_hidden, ca_mask = self._decode_step(
                    np.asarray([token_ids], dtype=np.int64), neighbour_hidden, neighbour_mask, past, ca_hidden,
                )
                decoder_ms.append((time.perf_counter() - started) * 1000)
                scores = logits[0, -1].astype(np.float64)
                if continuation is None:
                    next_token = int(np.argmax(scores))
                else:
                    next_token = int(target_values[step])
                    peak = float(np.max(scores))
                    nll += peak + math.log(float(np.exp(scores - peak).sum())) - float(scores[next_token])
                    scored += 1
                token_ids.append(next_token)
                if continuation is None and next_token == int(self.tokenizer.eos_token_id or 1):
                    break
                interval = int(settings["retrieval_interval"])
                if len(token_ids) >= int(settings["chunk_size"]) and len(token_ids) % interval == 0:
                    if schedule is None:
                        neighbours, diagnostics = self._retrieve(index_id, token_ids, settings)
                        retrieval_calls.append(diagnostics)
                    else:
                        neighbours, _ = schedule.boundary("", len(token_ids))
                    encoded = time.perf_counter()
                    neighbour_hidden, neighbour_mask = self._encode(neighbours, ca_hidden, ca_mask)
                    encoder_ms.append((time.perf_counter() - encoded) * 1000)
        finally:
            if schedule is not None:
                schedule.close()
        hits = []
        seen = set()
        for call in retrieval_calls:
            for hit in call.get("hits", []):
                key = (hit.get("source_id"), hit.get("chunk_id"))
                if key not in seen:
                    hits.append(hit)
                    seen.add(key)
        return {
            "token_ids": token_ids,
            "generated_ids": token_ids[len(initial_ids):],
            "nll": nll,
            "scored_tokens": scored,
            "evidence": hits,
            "diagnostics": {
                "prompt_tokens": len(initial_ids),
                "generated_tokens": len(token_ids) - len(initial_ids),
                "retrieval_calls": len(retrieval_calls),
                "retrieval_interval": int(settings["retrieval_interval"]),
                "stale_steps": int(settings["stale_steps"]),
                "dynamic_nprobe": bool(settings["dynamic_nprobe"]),
                "nprobe_values": [item.get("nprobe") for item in retrieval_calls],
                "embedding_latency_ms": sum(float(item.get("embedding_latency_ms", 0)) for item in retrieval_calls),
                "search_latency_ms": sum(float(item.get("search_latency_ms", 0)) for item in retrieval_calls),
                "wait_latency_ms": sum(float(item.get("wait_latency_ms", 0)) for item in retrieval_calls),
                "encoder_latency_ms": sum(encoder_ms),
                "decoder_latency_ms": sum(decoder_ms),
                "decoder_latency_per_token_ms": decoder_ms,
            },
        }

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.retrieval is None:
            raise RuntimeError("Inference service is not configured")
        settings = self._settings(request)
        if settings["chunk_size"] != 64:
            raise ValueError("Pinned RETRO/PipeRAG runtime requires chunk_size=64")
        prompt_ids = build_qa_prompt_ids(self.tokenizer, str(request.get("question") or ""), 64)
        started = time.perf_counter()
        result = self._run_tokens(str(request["index_id"]), prompt_ids, settings)
        elapsed_ms = (time.perf_counter() - started) * 1000
        decoded = self.tokenizer.decode(result["token_ids"], skip_special_tokens=True)
        generated = max(1, len(result["generated_ids"]))
        result["diagnostics"].update({
            "end_to_end_ms": elapsed_ms,
            "tokens_per_second": generated / max(elapsed_ms / 1000, 1e-9),
            "engine": "onnxruntime-cuda",
        })
        return {
            "prediction": extract_answer(decoded),
            "evidence": result["evidence"],
            "diagnostics": result["diagnostics"],
        }

    def score(self, index_id: str, tokens: list[int], settings: dict[str, Any]) -> dict[str, Any]:
        if len(tokens) <= 64:
            raise ValueError("Perplexity windows must contain more than 64 tokens")
        window = tokens[:64]
        continuation = tokens[64:]
        return self._run_tokens(index_id, window, settings, continuation=continuation)

    def score_request(self, request: dict[str, Any]) -> dict[str, Any]:
        tokens = [int(value) for value in request.get("token_ids") or []]
        if not tokens:
            tokens = self.tokenizer.encode(str(request.get("text") or ""), add_special_tokens=False)
        result = self.score(str(request["index_id"]), tokens, self._settings(request))
        scored = max(1, int(result["scored_tokens"]))
        return {
            "nll": float(result["nll"]), "scored_tokens": int(result["scored_tokens"]),
            "perplexity": math.exp(float(result["nll"]) / scored),
            "diagnostics": result["diagnostics"],
        }

    def profile_generation(self, request: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings({**request, "mode": "retro", "dynamic_nprobe": False})
        prompt = str(request.get("question") or "Retrieval augmented generation")
        ids = build_qa_prompt_ids(self.tokenizer, prompt, 64)
        runs = max(1, int(request.get("repetitions", 5)))
        warmups = max(0, int(request.get("warmups", 1)))
        observations = []
        nprobe_profile = self.retrieval.profile_nprobe({
            "index_id": str(request["index_id"]),
            "nprobes": list((self.configured.get("paper") or {}).get("nprobes", (1, 2, 4, 8, 16, 32, 64))),
            "repetitions": runs,
        })
        search_timings = {int(key): float(value) for key, value in (nprobe_profile.get("search_latency_ms") or {}).items()}
        per_probe = max((latency / nprobe for nprobe, latency in search_timings.items()), default=1e-6)
        for run in range(warmups + runs):
            result = self._run_tokens(str(request["index_id"]), ids, settings)
            if run >= warmups:
                observations.append(result["diagnostics"])
        maximum = max(len(item["decoder_latency_per_token_ms"]) for item in observations)
        decoder = []
        for index in range(maximum):
            values = [item["decoder_latency_per_token_ms"][min(index, len(item["decoder_latency_per_token_ms"]) - 1)] for item in observations]
            decoder.append(sum(values) / len(values))
        environment = self.environment()
        hardware_fingerprint = hashlib.sha256(json.dumps({
            "device": environment.get("device"),
            "onnx_providers": environment.get("onnx_providers"),
            "onnxruntime": environment.get("onnxruntime"),
            "model": self.model_fingerprint,
        }, sort_keys=True).encode()).hexdigest()
        profile = {
            "decoder_latency_ms": decoder,
            "encoder_latency_ms": sum(item["encoder_latency_ms"] for item in observations) / len(observations),
            "embedding_latency_ms": sum(item["embedding_latency_ms"] for item in observations) / len(observations),
            "search_latency_ms_per_nprobe": max(1e-6, per_probe),
            "search_latency_ms_by_nprobe": {str(key): value for key, value in search_timings.items()},
            "extra_overhead_ms": float((self.configured.get("piperag") or {}).get("extra_overhead_ms", 10.0)),
            "search_latency_budget_discount": float((self.configured.get("piperag") or {}).get("search_latency_budget_discount", 1.0)),
            "min_nprobe": int((self.configured.get("piperag") or {}).get("min_nprobe", 1)),
            "max_nprobe": int((self.configured.get("piperag") or {}).get("max_nprobe", 64)),
            "hardware_fingerprint": hardware_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "cache_reason": "rebuilt_for_current_hardware",
        }
        self.retrieval.set_profile({"index_id": str(request["index_id"]), "profile": profile})
        return profile

    def paper_evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        paper = dict(request.get("paper") or {})
        retro = dict(request.get("retro") or {})
        intervals = [int(value) for value in paper.get("intervals", (8, 16, 32, 64))]
        nprobes = [int(value) for value in paper.get("nprobes", (1, 2, 4, 8, 16, 32, 64))]
        max_windows = int(paper.get("max_windows_per_language", 32))
        window_tokens = int(paper.get("window_tokens", 1024))
        warmups = int(paper.get("latency_warmups", 1))
        repetitions = int(paper.get("latency_repetitions", 5))
        indexes = dict(request.get("indexes") or {})
        nlists = {key: int(value) for key, value in (request.get("nlists") or {}).items()}
        variants = set(request.get("variants") or ("retro", "piperag"))
        documents = list(request.get("documents") or [])
        document_datasets = {str(item.get("dataset") or "") for item in documents}
        document_datasets.discard("")
        dataset = str(request.get("dataset") or "")
        if not dataset and len(document_datasets) == 1:
            dataset = next(iter(document_datasets))
        if dataset not in {"its", "uniqa"}:
            raise ValueError(
                f"Paper evaluation requires dataset=its or dataset=uniqa; received {dataset!r}"
            )
        if document_datasets and document_datasets != {dataset}:
            raise ValueError(
                f"Paper documents do not match dataset={dataset}: found={sorted(document_datasets)}"
            )
        rows: list[dict[str, Any]] = []
        profiles: dict[str, Any] = {}
        for language in sorted({str(item.get("language") or "") for item in documents}):
            collection = collection_name(dataset, language)
            if collection not in indexes or collection not in nlists:
                raise ValueError(
                    "Paper index mapping is incomplete for "
                    f"dataset={dataset}, language={language}, collection={collection}: "
                    f"indexes={sorted(indexes)}, nlists={sorted(nlists)}. "
                    "Rebuild the paper index with --stage index or --stage all."
                )
            index_id = str(indexes[collection])
            valid_nprobes = [value for value in nprobes if value <= nlists.get(collection, value)]
            if not valid_nprobes:
                valid_nprobes = [1]
            windows: list[list[int]] = []
            for document in sorted((item for item in documents if item.get("language") == language), key=lambda item: str(item.get("id"))):
                tokens = self.tokenizer.encode(str(document.get("text") or ""), add_special_tokens=False)
                for offset in range(0, len(tokens), window_tokens):
                    window = tokens[offset: offset + window_tokens]
                    if len(window) > 64:
                        windows.append(window)
                    if len(windows) >= max_windows:
                        break
                if len(windows) >= max_windows:
                    break
            if not windows:
                raise ValueError(f"No held-out perplexity windows for language {language}")
            profile = self.profile_generation({
                "index_id": index_id,
                "question": self.tokenizer.decode(windows[0][:64], skip_special_tokens=True),
                "max_new_tokens": min(64, window_tokens - 64),
                "warmups": warmups,
                "repetitions": repetitions,
                "nprobe": min(16, max(valid_nprobes)),
            })
            profiles[language] = profile
            configurations = []
            if "retro" in variants:
                configurations.extend(("retro", 64, value, False) for value in valid_nprobes)
            if "piperag" in variants:
                configurations.extend(("piperag", interval, value, False) for interval in intervals for value in valid_nprobes)
                configurations.extend(("piperag", interval, int(retro.get("nprobe", 16)), True) for interval in intervals)
            for mode, interval, nprobe, dynamic in configurations:
                settings = self._settings({
                    "mode": mode,
                    "retrieval_interval": interval,
                    "stale_steps": interval if mode == "piperag" else 0,
                    "nprobe": nprobe,
                    "dynamic_nprobe": dynamic,
                    "max_new_tokens": max(1, window_tokens - 64),
                })
                total_nll = 0.0
                total_tokens = 0
                for window in windows:
                    scored = self.score(index_id, window, settings)
                    total_nll += float(scored["nll"])
                    total_tokens += int(scored["scored_tokens"])
                latency_values = []
                prefix = windows[0][:64]
                for repetition in range(warmups + repetitions):
                    started = time.perf_counter()
                    self._run_tokens(index_id, prefix, settings)
                    elapsed = (time.perf_counter() - started) * 1000
                    if repetition >= warmups:
                        latency_values.append(elapsed)
                effective_nprobe: int | str = "dynamic" if dynamic else nprobe
                rows.append({
                    "technique": mode,
                    "language": language,
                    "retrieval_interval": interval,
                    "stale_steps": interval if mode == "piperag" else 0,
                    "nprobe": effective_nprobe,
                    "dynamic_nprobe": dynamic,
                    "windows": len(windows),
                    "scored_tokens": total_tokens,
                    "perplexity": math.exp(total_nll / max(1, total_tokens)),
                    "latency_ms": sum(latency_values) / len(latency_values),
                    "latency_measurements_ms": latency_values,
                })
        return {"rows": rows, "profile": profiles}

    def unload(self) -> None:
        if self.retrieval is not None:
            self.retrieval.close()
        self.retrieval = None
        self.encoder_session = None
        self.decoder_session = None
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.bootstrap_device = ""
        self.configured = {}
        self.model_fingerprint = ""
        self.asset_manifest = {}
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
