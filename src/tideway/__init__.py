"""Phase-aware tiered memory management for MoE inference."""

from .backend import activate_backend

__version__ = "0.1.0"
__all__ = ["activate_backend", "__version__"]
