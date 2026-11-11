"""Build the baseline CUDA decoder for the selected device in the user cache."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from tideway._build import build_directory


def build_cuda_decoder(source):
    source = Path(source).resolve()
    compiler = os.environ.get("NVCC")
    if not compiler:
        toolkit = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if toolkit:
            compiler = str(Path(toolkit).expanduser() / "bin/nvcc")
        else:
            compiler = shutil.which("nvcc")
            if not compiler:
                compiler = next(
                    (
                        str(p)
                        for p in [
                            Path("/usr/local/cuda/bin/nvcc"),
                            *sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True),
                        ]
                        if p.is_file()
                    ),
                    None,
                )
    if not compiler or not shutil.which(compiler):
        raise FileNotFoundError("nvcc was not found; set CUDA_HOME or NVCC")
    architecture = os.environ.get("TIDEWAY_CUDA_ARCH")
    if not architecture:
        import torch

        major, minor = torch.cuda.get_device_capability()
        architecture = f"{major}{minor}"
    architecture = architecture.removeprefix("sm_").removeprefix("compute_").replace(".", "")
    if not re.fullmatch(r"\d{2,3}[a-z]?", architecture):
        raise ValueError("TIDEWAY_CUDA_ARCH must be a CUDA architecture such as 90 or 120")
    flags = ["-shared", "-Xcompiler", "-fPIC", "-O3", f"-arch=sm_{architecture}"]
    identity = source.read_bytes() + repr((compiler, flags)).encode()
    output_dir = build_directory()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"rans_cuda_{hashlib.sha256(identity).hexdigest()[:20]}.so"
    if output.is_file():
        return output
    fd, name = tempfile.mkstemp(prefix=".rans-cuda-", suffix=".so", dir=output_dir)
    os.close(fd)
    temporary = Path(name)
    try:
        subprocess.run([compiler, *flags, str(source), "-o", str(temporary)], check=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
