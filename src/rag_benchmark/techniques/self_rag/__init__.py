"""Configuration is lightweight; load the pipeline only when requested."""
from .config import SelfRagConfig, VARIANTS

__all__ = ["SelfRagConfig", "VARIANTS", "SelfRagPipeline"]


def __getattr__(name):
    if name == "SelfRagPipeline":
        from .pipeline import SelfRagPipeline
        return SelfRagPipeline
    raise AttributeError(name)
