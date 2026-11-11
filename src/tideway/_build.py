"""Build small native helpers in a user-writable cache."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import shutil
import site
import subprocess
import tempfile


def build_directory() -> Path:
    configured = os.environ.get("TIDEWAY_BUILD_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (cache.expanduser() / "tideway" / "native").resolve()


def cuda_include_directory() -> Path:
    """Find driver headers without importing torch or initializing CUDA."""
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured:
        roots = [Path(configured).expanduser()]
    else:
        roots = []
        nvcc = shutil.which("nvcc")
        if nvcc:
            roots.append(Path(nvcc).resolve().parent.parent)
        roots.extend([Path("/usr/local/cuda"), Path("/opt/cuda")])
        roots.extend(sorted(Path("/usr/local").glob("cuda-*"), reverse=True))
        for package_dir in site.getsitepackages() + [site.getusersitepackages()]:
            roots.append(Path(package_dir) / "nvidia" / "cuda_runtime")
    for root in roots:
        includes = [root / "include", *sorted((root / "targets").glob("*/include"))]
        for include in includes:
            if (include / "cuda.h").is_file():
                return include.resolve()
    raise FileNotFoundError(
        "CUDA driver headers were not found; set CUDA_HOME to the CUDA toolkit directory"
    )


def build_shared_library(
    source: Path,
    *,
    cuda_headers: bool = False,
    libraries: tuple[str, ...] = ("-ldl",),
    optimization: str = "-O2",
) -> Path:
    source = Path(source).resolve()
    compiler = shlex.split(os.environ.get("CC", "gcc"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise FileNotFoundError("C compiler was not found; install GCC or set CC")
    flags = ["-std=gnu11", "-shared", "-fPIC", optimization, "-Wall", "-Wextra"]
    if cuda_headers:
        flags.extend(["-I", str(cuda_include_directory())])
    identity = source.read_bytes() + repr((compiler, flags, libraries)).encode()
    digest = hashlib.sha256(identity).hexdigest()[:20]
    directory = build_directory()
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{source.stem}_{digest}.so"
    if output.is_file():
        return output
    fd, temporary_name = tempfile.mkstemp(prefix=f".{source.stem}-", suffix=".so", dir=directory)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        subprocess.run(
            [*compiler, *flags, "-o", str(temporary), str(source), *libraries], check=True
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
