"""Select an assembled Tideway backend before importing FreeToken."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


def activate_backend(path: str | Path | None = None) -> Path:
    """Prepend the backend directory to sys.path; refuse an already loaded foreign backend."""
    configured = path or os.environ.get("TIDEWAY_BACKEND")
    if configured:
        backend = Path(configured).expanduser().resolve()
    else:
        checkout = Path(__file__).resolve().parents[2]
        if not (checkout / "backends" / "freetoken_tideway" / "changes.json").is_file():
            raise FileNotFoundError(
                "Set TIDEWAY_BACKEND or pass the assembled backend directory explicitly"
            )
        backend = checkout / "build" / "freetoken_tideway"
    package = backend / "freetoken"
    if not (package / "__init__.py").is_file() or not (backend / "tideway-backend.json").is_file():
        raise FileNotFoundError(
            f"{backend} is not an assembled Tideway backend; run scripts/build_backend.py"
        )
    loaded = sys.modules.get("freetoken")
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(package.resolve()):
            raise RuntimeError(
                "Call activate_backend() before importing FreeToken; restart this Python process"
            )
    location = str(backend)
    if location in sys.path:
        sys.path.remove(location)
    sys.path.insert(0, location)
    importlib.invalidate_caches()
    return backend
