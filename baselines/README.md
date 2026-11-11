# Baselines

The baseline adapters use the installed FreeToken 0.1.2 package. Start them in
a fresh Python process. `tideway.activate_backend()` selects the Tideway source
overlay and belongs to Tideway execution.

| System | Implementation |
|---|---|
| FreeToken | Native FreeToken 0.1.2 |
| FreeToken+SSD | GPU LRU, independent pinned-RAM LRU and demand SSD reads |
| SSD-LLaMA | Aligned rANS records, compressed RAM cache and frequency-recency GPU retention |
| FlashMoE | Trained per-layer next-use predictor, bounded RAM LRU and raw checkpoint reads |
| llama.cpp | Pinned upstream runtime with optional RSS governor and token replay libraries |

SSD-LLaMA and FlashMoE are independent implementations of their published
mechanisms. Their model computation uses FreeToken kernels. Shared code covers
checkpoint geometry, transport and bookkeeping.

## FreeToken and FreeToken+SSD

`tideway.baselines.create_freetoken(model, **kwargs)` constructs the native
FreeToken `LLM` with the caller's engine settings. All native keyword arguments
are passed through.

For an initialized offload engine, attach FreeToken+SSD as follows:

```python
from tideway.offload import DirectExperts, from_checkpoint
from tideway.baselines import install_freetoken_ssd

geometry = from_checkpoint(model_directory)
reader = DirectExperts(model_directory, geo=geometry, workers=1)
session = install_freetoken_ssd(
    llm.engine.moe_offload_cache,
    geo=geometry,
    host_rows=host_rows,
    reader=reader,
)
```

The application supplies the initialized `llm`, checkpoint directory and expert
capacity. GPU cache slots must hold two prefill buffers; RAM must hold at least
one expert layer. Configure SSD adapters with `moe_backend="offload"`,
`cuda_graph_bs=[]` and `moe_prefill_overlap=False` before installing their hooks.
The default SSD reader is single-worker; another concurrency requires an
explicit matching `ssd_workers`.
GPU eviction never writes experts back into RAM. After stopping inference,
`session.uninstall()` restores the cache and layer hooks; close the caller-owned
reader afterwards.

`tideway.baselines.gdn_carry.install()` carries pending Qwen GDN snapshots across
chunk boundaries. It is an explicit shared correctness hook for applications
that use the corresponding chunked FreeToken scheduler.

## SSD-LLaMA and FlashMoE

Prepare the aligned expert pack and verified metadata:

```bash
python scripts/prepare_pack.py --model /path/to/checkpoint --out /path/to/pack
```

`--sample` prepares nine experts for a format check. Actual inference needs a
complete pack. SSD-LLaMA reads compressed records from `experts.pack`; FlashMoE
uses the verified manifest and reads raw expert bytes from the checkpoint.

Attach either adapter to an initialized FreeToken offload cache:

```python
from tideway.baselines import install_ssd_llama, install_flashmoe

session = install_ssd_llama(
    llm.engine.moe_offload_cache,
    geo=geometry,
    pack=pack_directory,
    host_rows=host_rows,
)

# In a separate run, select the learned policy:
session = install_flashmoe(
    llm.engine.moe_offload_cache,
    geo=geometry,
    pack=pack_directory,
    predictor=predictor_directory,
    host_rows=host_rows,
)
```

These adapters use eager execution and return a session with `.manager` and
`.uninstall()`. Their internally created readers close with the session. Optional
`host_budget_bytes` accounts for aligned host records and reader workspace;
`lanes` controls concurrent delivery. FlashMoE GPU slots divide evenly by the
number of expert layers.

SSD-LLaMA orders its GPU retention score lexicographically by frequency and
recency, retains existing host copies on eviction and discards GPU-only victims.
BF16 records use byte-plane encoding; FP4 records preserve packed bytes and scales.
FlashMoE uses independent per-layer three-layer networks with 128-wide hidden
layers and SiLU activation. Both pagers use bounded concurrent O_DIRECT reads.

Train predictors from calibration routing records:

```bash
python scripts/train_flashmoe.py --routes /path/to/routes.json --out /path/to/predictor \
    --experts 256 --layers 43 --capacity 128
```

The JSON is a list of records shaped as
`{"case": "calibration_0", "layer": 0, "ids": [[1, 7], [3, 5]]}`.
Supply at least three distinct calibration cases and all model layers. Earlier
cases train the networks and the final case validates them. The output contains
`predictor.pt` and a manifest recording the source hash and training settings.

## llama.cpp

Build commit `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb` and the preload libraries:

```bash
baselines/llama_cpp/build.sh
```

This requires Git, CMake 3.24 or newer, GCC/G++ and the CUDA toolkit. CMake selects
the installed GPU architecture by default. Set `TIDEWAY_CUDA_ARCH=120` or another
architecture to override it; `GGML_CUDA=OFF` selects a CPU build. `JOBS` controls
build parallelism. The build directory is an optional positional argument.

Start the server with a GGUF model and mmap-weight RSS governor:

```bash
python scripts/run_llama.py --model /path/to/model.gguf --ram-gib 32 \
    --out runs/llama -- --host 127.0.0.1 --port 8080 -ngl 99
```

The governor reclaims mapped GGUF windows using `madvise` and `posix_fadvise`.
Its logs and control files live under `--out`. `--headroom-mib` sets its reserve.
Optional `TIDEWAY_LLAMA_CPUS` and `TIDEWAY_LLAMA_MONITOR_CPU` select governor CPU
affinity; otherwise it inherits the process affinity.

`libllama_replay.so` preserves the native sampler call and optionally substitutes
tokens from a shared control buffer. Its environment variables are
`TIDEWAY_REPLAY_CONTROL`, `TIDEWAY_REPLAY_TOKEN_EVENTS` and
`TIDEWAY_REPLAY_BATCH_EVENTS`. Normal server generation uses its native sampler.

## Native compilation

The rANS CPU codec and CUDA decoder compile into the same user cache as Tideway's
native helpers. `TIDEWAY_BUILD_DIR` overrides the cache; `CC` selects the C compiler.
`CUDA_HOME` or `NVCC` selects the CUDA compiler. The decoder detects the active GPU
architecture unless `TIDEWAY_CUDA_ARCH` is set explicitly.
