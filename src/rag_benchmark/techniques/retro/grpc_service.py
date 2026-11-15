from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


def json_handler(function: Callable[[dict[str, Any]], dict[str, Any]]):
    import grpc

    def invoke(raw: dict[str, Any], context):
        try:
            return function(raw)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"{type(exc).__name__}: {exc}")
            return {"protocol_version": raw.get("protocol_version", 1), "status": "error", "error": str(exc)}

    return grpc.unary_unary_rpc_method_handler(
        invoke,
        request_deserializer=lambda value: json.loads(value.decode("utf-8")),
        response_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
    )


def json_stream_handler(function: Callable):
    import grpc

    def invoke(raw_values, context):
        try:
            return function(raw_values)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"{type(exc).__name__}: {exc}")
            return {"protocol_version": 1, "status": "error", "error": str(exc)}

    return grpc.stream_unary_rpc_method_handler(
        invoke,
        request_deserializer=lambda value: json.loads(value.decode("utf-8")),
        response_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
    )


def create_json_server(
    service_name: str, methods: dict[str, Callable], host: str, port: int,
    max_message_mb: int = 64, stream_methods: dict[str, Callable] | None = None,
):
    import grpc

    size = max_message_mb * 1024 * 1024
    server = grpc.server(
        ThreadPoolExecutor(max_workers=8),
        options=(("grpc.max_send_message_length", size), ("grpc.max_receive_message_length", size)),
    )
    handlers = {name: json_handler(function) for name, function in methods.items()}
    handlers.update({name: json_stream_handler(function) for name, function in (stream_methods or {}).items()})
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(service_name, handlers),))
    bound_port = server.add_insecure_port(f"{host}:{port}")
    return server, bound_port


def serve_json(
    service_name: str, methods: dict[str, Callable], host: str, port: int,
    max_message_mb: int = 64, stream_methods: dict[str, Callable] | None = None,
):
    server, _ = create_json_server(service_name, methods, host, port, max_message_mb, stream_methods)
    server.start()
    server.wait_for_termination()
