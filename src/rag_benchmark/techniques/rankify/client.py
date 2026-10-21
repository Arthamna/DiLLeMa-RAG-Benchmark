from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


class RankifyWorkerClient:
    def __init__(self, python: Path):
        self.python = python

    def invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.python.is_file():
            raise FileNotFoundError(
                f"Rankify worker Python not found: {self.python}. Run scripts/setup_rankify_ubuntu.sh"
            )
        worker_env = os.environ.copy()
        worker_env["PATH"] = os.pathsep.join([
            str(self.python.parent),
            worker_env.get("PATH", ""),
        ])

        completed = subprocess.run(
            [str(self.python), "-m", "rag_benchmark_rankify.worker"],
            input=json.dumps(request, ensure_ascii=False) + "\n",
            text=True,
            capture_output=True,
            check=False,
            env=worker_env,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Rankify worker failed ({completed.returncode}): {completed.stderr.strip() or completed.stdout.strip()}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("Rankify worker returned no JSON response")
        response = json.loads(lines[-1])
        if response.get("status") != "ok":
            raise RuntimeError(f"Rankify worker error: {response.get('error', 'unknown error')}")
        return response
