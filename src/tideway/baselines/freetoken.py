"""Native FreeToken 0.1.2 without the Tideway source overlay."""

from pathlib import Path

from ._backend import require_pristine_backend


def create_llm(model, **kwargs):
    require_pristine_backend()
    from freetoken.llm import LLM

    return LLM(str(Path(model).expanduser().resolve()), **kwargs)
