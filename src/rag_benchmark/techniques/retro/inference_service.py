from __future__ import annotations

import argparse
import os
import threading
from typing import Any

from .grpc_service import serve_json
from .onnx_runtime import OnnxRetroRuntime


class InferenceService:
    def __init__(self, protocol_version: int = 1):
        self.protocol_version = protocol_version
        self.runtime = OnnxRetroRuntime()
        self.lock = threading.RLock()

    def _check(self, request: dict[str, Any]) -> None:
        if int(request.get("protocol_version", -1)) != self.protocol_version:
            raise ValueError("protocol version mismatch")

    def _response(self, **value):
        return {"protocol_version": self.protocol_version, "status": "ok", **value}

    def health(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        environment = self.runtime.environment()
        if os.getenv("RETRO_REQUIRE_CUDA", "1") != "0" and (
            not environment.get("cuda") or "CUDAExecutionProvider" not in environment.get("onnx_providers", [])
        ):
            return {
                "protocol_version": self.protocol_version, "status": "error",
                "error": "CUDA and CUDAExecutionProvider are required", "environment": environment,
            }
        return self._response(
            role="inference",
            lazy_loaded=self.runtime.model is not None,
            model_fingerprint=self.runtime.model_fingerprint or None,
            onnx=self.runtime.asset_manifest or None,
            environment=environment,
            image_version=os.getenv("BENCHMARK_IMAGE_VERSION", "development"),
        )

    def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        with self.lock:
            configured = self.runtime.configure(request)
        return self._response(**configured)

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        with self.lock:
            result = self.runtime.generate(request)
        return self._response(**result)

    def score_sequence(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        with self.lock:
            result = self.runtime.score_request(request)
        return self._response(**result)

    def profile_generation(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        with self.lock:
            profile = self.runtime.profile_generation(request)
        return self._response(profile=profile)

    def paper_evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        with self.lock:
            result = self.runtime.paper_evaluate(request)
        return self._response(**result)

    def unload(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check(request)
        with self.lock:
            self.runtime.unload()
        return self._response(unloaded=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--protocol-version", type=int, default=1)
    parser.add_argument("--max-message-mb", type=int, default=64)
    args = parser.parse_args(argv)
    service = InferenceService(args.protocol_version)
    serve_json("rag.InferenceService", {
        "Health": service.health,
        "Configure": service.configure,
        "Generate": service.generate,
        "ScoreSequence": service.score_sequence,
        "ProfileGeneration": service.profile_generation,
        "PaperEvaluate": service.paper_evaluate,
        "Unload": service.unload,
    }, args.host, args.port, args.max_message_mb)


if __name__ == "__main__":
    main()
