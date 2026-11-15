from __future__ import annotations

import sys

from .client import InferenceServiceClient, RetrievalServiceClient
from .config import ServiceConfig


def main(argv=None):
    values = list(argv or sys.argv[1:])
    if len(values) != 2 or values[0] not in {"retrieval", "inference"}:
        raise SystemExit("usage: healthcheck retrieval|inference HOST:PORT")
    role, endpoint = values
    config = ServiceConfig(
        retrieval_endpoint=endpoint if role == "retrieval" else "unused:0",
        inference_endpoint=endpoint if role == "inference" else "unused:0",
        timeout_seconds=3,
    )
    client = RetrievalServiceClient(config) if role == "retrieval" else InferenceServiceClient(config)
    try:
        response = client.health()
    finally:
        client.close()
    if response.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
