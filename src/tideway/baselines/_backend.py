"""Require the installed FreeToken backend for baseline execution."""

from importlib import metadata, util
from pathlib import Path


def require_pristine_backend() -> Path:
    distribution = metadata.distribution("freetoken")
    if distribution.version != "0.1.2":
        raise RuntimeError("Baselines require freetoken==0.1.2")
    expected = Path(distribution.locate_file("freetoken/__init__.py")).resolve()
    spec = util.find_spec("freetoken")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
        raise RuntimeError(
            "Run baselines in a fresh process with the installed FreeToken backend; "
            "activate_backend() selects the Tideway backend"
        )
    return expected.parent
