from __future__ import annotations

import logging
import os

LOGGER = logging.getLogger(__name__)


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


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
