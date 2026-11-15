from __future__ import annotations

import json
import time
from typing import Any, Callable

from .config import ServiceConfig


class ServiceError(RuntimeError):
    pass


class JsonGrpcClient:
    """Small JSON codec over gRPC's unary transport.

    JSON keeps the protocol inspectable while gRPC still supplies deadlines,
    HTTP/2 framing, and a stable process/host boundary.
    """

    def __init__(
        self, endpoint: str, service: str, config: ServiceConfig,
        transport: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
    ):
        self.endpoint = endpoint
        self.service = service
        self.config = config
        self._transport = transport
        self._channel = None

    def _grpc_transport(self, method: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            import grpc
        except ImportError as exc:
            raise ServiceError("grpcio is required for RETRO/PipeRAG service calls") from exc
        if self._channel is None:
            size = self.config.max_message_mb * 1024 * 1024
            self._channel = grpc.insecure_channel(
                self.endpoint,
                options=(("grpc.max_send_message_length", size), ("grpc.max_receive_message_length", size)),
            )
        call = self._channel.unary_unary(
            f"/{self.service}/{method}",
            request_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
            response_deserializer=lambda value: json.loads(value.decode("utf-8")),
        )
        return call(payload, timeout=timeout)

    def call(self, method: str, payload: dict[str, Any], *, retry: bool = True) -> dict[str, Any]:
        request = {"protocol_version": self.config.protocol_version, **payload}
        transport = self._transport or self._grpc_transport
        attempts = 2 if retry else 1
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                response = transport(method, request, self.config.timeout_seconds)
                if int(response.get("protocol_version", -1)) != self.config.protocol_version:
                    raise ServiceError(f"{self.service} protocol version mismatch")
                if response.get("status") == "error":
                    raise ServiceError(str(response.get("error") or f"{self.service} request failed"))
                return response
            except Exception as exc:
                last = exc
                if attempt + 1 < attempts:
                    time.sleep(0.1)
        raise ServiceError(f"{self.service}.{method} failed after {attempts} attempt(s): {last}") from last

    def stream_call(self, method: str, payloads, *, retry: bool = False) -> dict[str, Any]:
        values = [{"protocol_version": self.config.protocol_version, **payload} for payload in payloads]
        if self._transport is not None:
            merged = dict(values[0])
            merged["documents"] = [document for value in values for document in value.get("documents", [])]
            return self.call(method, merged, retry=retry)
        try:
            import grpc
        except ImportError as exc:
            raise ServiceError("grpcio is required for RETRO/PipeRAG service calls") from exc
        if self._channel is None:
            size = self.config.max_message_mb * 1024 * 1024
            self._channel = grpc.insecure_channel(
                self.endpoint,
                options=(("grpc.max_send_message_length", size), ("grpc.max_receive_message_length", size)),
            )
        call = self._channel.stream_unary(
            f"/{self.service}/{method}",
            request_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
            response_deserializer=lambda value: json.loads(value.decode("utf-8")),
        )
        response = call(iter(values), timeout=self.config.timeout_seconds)
        if int(response.get("protocol_version", -1)) != self.config.protocol_version:
            raise ServiceError(f"{self.service} protocol version mismatch")
        if response.get("status") == "error":
            raise ServiceError(str(response.get("error") or f"{self.service} request failed"))
        return response

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()


class RetrievalServiceClient:
    def __init__(self, config: ServiceConfig, transport=None):
        self.rpc = JsonGrpcClient(config.retrieval_endpoint, "rag.RetrievalService", config, transport)

    def health(self) -> dict[str, Any]:
        return self.rpc.call("Health", {}, retry=False)

    def build_index(self, payload: dict[str, Any]) -> dict[str, Any]:
        value = dict(payload)
        documents = list(value.pop("documents", []))
        batches = []
        for offset in range(0, len(documents), 128):
            batches.append({**(value if offset == 0 else {"index_id": value.get("index_id")}),
                            "documents": documents[offset: offset + 128]})
        if not batches:
            batches = [{**value, "documents": []}]
        return self.rpc.stream_call("BuildIndex", batches, retry=False)

    def load_index(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.rpc.call("LoadIndex", payload, retry=False)

    def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.rpc.call("Retrieve", payload)

    def set_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.rpc.call("SetProfile", payload, retry=False)

    def set_nprobe(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.rpc.call("SetNprobe", payload, retry=False)

    def profile_nprobe(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.rpc.call("ProfileNprobe", payload, retry=False)

    def close(self) -> None:
        self.rpc.close()


class InferenceServiceClient:
    def __init__(self, config: ServiceConfig, transport=None):
        self.rpc = JsonGrpcClient(config.inference_endpoint, "rag.InferenceService", config, transport)

    def health(self) -> dict[str, Any]:
        return self.rpc.call("Health", {}, retry=False)

    def configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.rpc.call("Configure", payload, retry=False)

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.rpc.call("Generate", payload)

    def score_sequence(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.rpc.call("ScoreSequence", payload)

    def paper_evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.rpc.call("PaperEvaluate", payload, retry=False)

    def profile_generation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.rpc.call("ProfileGeneration", payload, retry=False)

    def unload(self) -> dict[str, Any]:
        return self.rpc.call("Unload", {}, retry=False)

    def close(self) -> None:
        self.rpc.close()
