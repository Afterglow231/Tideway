"""Baseline adapters and independent SSD-LLaMA/FlashMoE implementations."""

from importlib import import_module

_EXPORTS = {
    "create_freetoken": (".freetoken", "create_llm"),
    "install_freetoken_ssd": (".freetoken_ssd", "install"),
    "install_ssd_llama": (".ssd_llama", "install"),
    "install_flashmoe": (".flashmoe", "install"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
