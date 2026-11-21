"""Isolated RankLLM runtime. Invoked using the worker virtualenv's Python."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


LOGGER = logging.getLogger("rank_zephyr.worker")
WINDOW = 20
STRIDE = 10
CONTEXT_SIZE = 4096


def _probe() -> dict:
    import torch
    from rank_llm.rerank.listwise.rank_listwise_os_llm import RankListwiseOSLLM
    from vllm.model_executor.layers.quantization import get_quantization_config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable in .venv-rank-llm")
    gpu = torch.cuda.get_device_properties(0)
    get_quantization_config("bitsandbytes")
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    return {
        "rank_llm": version("rank-llm"), "vllm": version("vllm"),
        "bitsandbytes": version("bitsandbytes"), "torch": version("torch"),
        "cuda": torch.version.cuda, "gpu": gpu.name, "gpu_bytes": gpu.total_memory,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "torch_architectures": torch.cuda.get_arch_list(),
        "adapter": RankListwiseOSLLM.__name__,
    }


def _snapshot(model: str, revision: str, local_path: str) -> Path:
    if local_path:
        LOGGER.info("Using local RankZephyr snapshot %s", local_path)
        snapshot = Path(local_path).resolve(strict=True)
    else:
        from huggingface_hub import snapshot_download

        LOGGER.info("Resolving RankZephyr checkpoint %s at %s", model, revision)
        snapshot = Path(snapshot_download(
            model, revision=revision, allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
        ))
    if not (snapshot / "config.json").is_file() or not (snapshot / "tokenizer_config.json").is_file():
        raise ValueError(f"Incomplete RankZephyr config/tokenizer in {snapshot}")
    if not any((snapshot / name).is_file() for name in ("tokenizer.model", "tokenizer.json")):
        raise ValueError(f"RankZephyr tokenizer files are missing in {snapshot}")
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        if not shards or any(not (snapshot / name).is_file() for name in shards):
            raise ValueError(f"RankZephyr safetensors shards are missing in {snapshot}")
    elif not (snapshot / "model.safetensors").is_file():
        raise ValueError(f"RankZephyr safetensors weights are missing in {snapshot}")
    if snapshot.parent.name == "snapshots" and not snapshot.name.startswith(revision):
        raise ValueError(f"RankZephyr snapshot {snapshot.name} differs from requested revision {revision}")
    return snapshot


def _stop_server(process) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    # Engine children can still be present after the API process exits.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


@contextmanager
def _server(snapshot: Path, log_path: Path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}/v1"
    command = [
        sys.executable, "-I", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(snapshot), "--served-model-name", str(snapshot),
        "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
        "--quantization", "bitsandbytes", "--load-format", "bitsandbytes",
        "--max-model-len", str(CONTEXT_SIZE), "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.80", "--enforce-eager",
        "--generation-config", "vllm",
    ]
    opener = build_opener(ProxyHandler({}))
    with log_path.open("w", encoding="utf-8") as log:
        LOGGER.info("Starting 4-bit vLLM. Server log: %s", log_path)
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited with code {process.returncode}; see {log_path}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"vLLM was not ready after 600 seconds; see {log_path}")
                try:
                    with opener.open(f"{base_url}/models", timeout=2) as response:
                        models = json.load(response)["data"]
                except (URLError, TimeoutError):
                    time.sleep(1)
                    continue
                if not any(model["id"] == str(snapshot) for model in models):
                    raise RuntimeError(f"Unexpected model on the RankZephyr vLLM endpoint; see {log_path}")
                break
            yield base_url
        finally:
            _stop_server(process)


def _ranker(snapshot: Path, base_url: str):
    from rank_llm.rerank.listwise.rank_listwise_os_llm import RankListwiseOSLLM

    class ExactChatBudget(RankListwiseOSLLM):
        def get_num_tokens(self, prompt):
            if isinstance(prompt, list):
                rendered = self._tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True,
                )
                return len(self._tokenizer.encode(rendered, add_special_tokens=False))
            return super().get_num_tokens(prompt)

    return ExactChatBudget(
        model=str(snapshot), base_url=base_url, context_size=CONTEXT_SIZE, device="cuda",
        window_size=WINDOW, stride=STRIDE, batch_size=1, use_logits=False,
        sampling_kwargs={"temperature": 0.0},
    )


def _window_lengths(count: int) -> list[int]:
    lengths = []
    end = count
    while end > 0:
        start = max(0, end - WINDOW)
        lengths.append(end - start)
        if start == 0:
            break
        end -= STRIDE
    return lengths


async def _rank(ranker, row: dict) -> dict:
    candidates = row["candidates"]
    if not candidates:
        return {"ranked_ids": [], "windows": 0, "repaired_windows": 0}
    from rank_llm.data import Candidate, Query, Request

    if ranker.get_num_tokens(row["question"]) > CONTEXT_SIZE // 4:
        raise ValueError("Question exceeds the RankZephyr prompt budget")
    request = Request(
        query=Query(text=row["question"], qid=row["id"]),
        candidates=[Candidate(
            docid=item["id"], score=item["score"], doc={"text": item["text"]},
        ) for item in candidates],
    )
    result, = await ranker.rerank_batch_async(
        [request], rank_start=0, rank_end=len(candidates), top_k_retrieve=len(candidates),
        shuffle_candidates=False, populate_invocations_history=True,
    )
    ranked = [item.docid for item in result.candidates]
    expected = [item["id"] for item in candidates]
    if result.query.qid != row["id"] or sorted(ranked) != sorted(expected) or len(set(ranked)) != len(ranked):
        raise ValueError("RankLLM returned an invalid candidate permutation")
    lengths = _window_lengths(len(expected))
    history = result.invocations_history
    if len(history) != len(lengths):
        raise RuntimeError(f"RankLLM completed {len(history)} of {len(lengths)} expected windows")
    repairs = []
    for number, (invocation, size) in enumerate(zip(history, lengths), 1):
        if not invocation.token_usage:
            raise RuntimeError(f"vLLM request failed: {invocation.response}")
        response = (invocation.response or "").strip()
        ids = [int(item) for item in re.findall(r"\[(\d+)\]", response)]
        if not re.fullmatch(r"\[\d+\](\s*>\s*\[\d+\])*", response) or sorted(ids) != list(range(1, size + 1)):
            repairs.append({"window": number, "response": response})
    return {
        "ranked_ids": ranked, "windows": len(history),
        "repaired_windows": len(repairs), "repairs": repairs,
    }


async def _process_requests(ranker, input_path: Path, output_path: Path) -> None:
    # Keep the OpenAI client's pooled connections on one event loop.
    with input_path.open(encoding="utf-8") as inputs, output_path.open("w", encoding="utf-8") as outputs:
        for number, line in enumerate(inputs, 1):
            row = json.loads(line)
            started = time.perf_counter()
            try:
                result = await _rank(ranker, row)
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            result.update(id=row["id"], rerank_latency_ms=(time.perf_counter() - started) * 1000)
            outputs.write(json.dumps(result) + "\n")
            outputs.flush()
            LOGGER.info("RankZephyr rerank %s %s: %s", number, row["id"],
                        result.get("error") or f"{result['windows']} windows")


def _exit_on_parent_death(parent_pid: int) -> None:
    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG)")
    if os.getppid() != parent_pid:
        raise RuntimeError("Benchmark parent exited before RankZephyr startup")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--server-log", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.probe:
        print(json.dumps(_probe()))
        return

    _exit_on_parent_death(args.parent_pid)
    diagnostics = {"status": "starting", "model": args.model, "revision": args.revision}
    try:
        diagnostics.update(_probe())
        snapshot = _snapshot(args.model, args.revision, args.snapshot)
        diagnostics.update({
            "snapshot": str(snapshot), "snapshot_source": "local" if args.snapshot else "huggingface",
            "quantization": "bitsandbytes-4bit",
            "compute_dtype": "bfloat16", "context_size": CONTEXT_SIZE,
            "window": WINDOW, "stride": STRIDE, "temperature": 0.0,
            "rank_llm_commit": "0fd5678ffc43488531ba26a42f367d42480a75ec",
        })
        with _server(snapshot, args.server_log) as base_url:
            ranker = _ranker(snapshot, base_url)
            try:
                asyncio.run(_process_requests(ranker, args.input, args.output))
            finally:
                ranker.close()
        diagnostics["status"] = "complete"
    except Exception as exc:
        diagnostics.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.diagnostics.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
