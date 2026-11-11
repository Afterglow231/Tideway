# Runtime guide

[Back to Tideway](../README.md)

## Installation

Use Linux, Python 3.12 or newer, an NVIDIA GPU, a CUDA 13 toolkit and GCC.
Install from the source checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/build_backend.py
```

The build script checks upstream and Tideway file hashes, then assembles
`build/freetoken_tideway/`. The installed FreeToken package stays intact.

Set `CUDA_HOME` to your toolkit directory when it is outside the standard
locations. Native C helpers compile on demand into
`$XDG_CACHE_HOME/tideway/native` or `~/.cache/tideway/native`.
`TIDEWAY_BUILD_DIR` overrides that cache directory; `CC` selects the C compiler.

## Backend selection

Activate the assembled backend before importing FreeToken:

```python
from tideway import activate_backend

activate_backend()

from freetoken.llm import LLM
```

For a backend built elsewhere, pass its directory to `activate_backend(path)`
or set `TIDEWAY_BACKEND`. The directory must contain `freetoken/` and
`tideway-backend.json`. Model checkpoints are supplied by the caller; the
runtime does not assume a model directory or set CPU affinity and thread counts.

## Runtime integration

For an initialized FreeToken engine with an offload cache, create the checkpoint
reader and install the expert tier before starting inference:

```python
from tideway.offload import DirectExperts, from_checkpoint, install

geometry = from_checkpoint(model_directory)
reader = DirectExperts(model_directory, geo=geometry, workers=8)
session = install(
    llm.engine.moe_offload_cache,
    geo=geometry,
    host_rows=host_rows,
    reader=reader,
    arm="tideway",
)
```

Here `llm`, `model_directory` and `host_rows` are supplied by the application.
`host_rows` counts expert slots in pinned RAM. The engine's GPU expert slots,
context length and prefill workspace must be configured for the device.
`arm="tideway_hybrid"` selects the static GPU-core layout and accepts a routing
histogram through `prior`. `tideway.policy.choose_memory_policy` selects a
layout from the declared capacities. The reader remains application-owned.
Stop inference and drain manager-owned work before uninstalling the session
and closing the reader.

## More

- [Baseline APIs and tools](../baselines/README.md)
