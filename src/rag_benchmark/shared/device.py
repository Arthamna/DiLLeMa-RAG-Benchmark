from __future__ import annotations

import logging
import os

LOGGER = logging.getLogger(__name__)


class CUDAStageError(RuntimeError):
    """CUDA failed at a named synchronous pipeline boundary."""


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def synchronize_cuda(device: object, stage: str) -> None:
    """Expose asynchronous CUDA failures at the operation that caused them."""

    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception as exc:
        raise CUDAStageError(
            f"CUDA failure during {stage}: {type(exc).__name__}: {exc}"
        ) from exc


def is_fatal_accelerator_error(exc: BaseException) -> bool:
    """Return whether an exception means the current CUDA process must stop."""

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, CUDAStageError):
            return True
        message = f"{type(current).__name__}: {current}".casefold()
        if any(token in message for token in (
            "cuda error",
            "cuda_error_",
            "cuda failure",
            "device-side assert",
            "cublas_status",
            "cudnn_status",
        )):
            return True
        current = current.__cause__ or current.__context__
    return False


def resolve_torch_device(component: str, requested: str | None = None) -> str:
    """Prefer CUDA, but keep the pipeline runnable on CPU with a warning."""

    device = (requested or os.getenv("RAG_DEVICE", "auto")).strip().casefold()
    if device == "auto":
        if cuda_available():
            device = "cuda"
        else:
            LOGGER.warning(
                "CUDA GPU tidak terdeteksi untuk %s; evaluasi tetap dilanjutkan menggunakan CPU.",
                component,
            )
            return "cpu"
    elif device.startswith("cuda") and not cuda_available():
        LOGGER.warning(
            "CUDA GPU tidak terdeteksi untuk %s; evaluasi tetap dilanjutkan menggunakan CPU.",
            component,
        )
        return "cpu"

    if device.startswith("cuda"):
        import torch

        index = int(device.split(":", 1)[1]) if ":" in device else 0
        LOGGER.info("%s menggunakan GPU CUDA: %s", component, torch.cuda.get_device_name(index))
    elif device == "cpu":
        LOGGER.info("%s dikonfigurasi menggunakan CPU.", component)
    else:
        LOGGER.info("%s menggunakan device '%s'.", component, device)
    return device
