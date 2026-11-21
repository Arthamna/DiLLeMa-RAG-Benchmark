"""Configuration is lightweight; load the pipeline only when requested."""
from .config import BgeM3Config, VARIANTS

__all__ = ["BgeM3Config", "VARIANTS", "BgeM3Pipeline"]


def __getattr__(name):
    if name == "BgeM3Pipeline":
        from .pipeline import BgeM3Pipeline
        return BgeM3Pipeline
    raise AttributeError(name)
