"""SSD-LLaMA: compressed RAM records and frequency-recency GPU retention."""

from pathlib import Path


def install(cache, *, geo, pack, host_rows, **kwargs):
    from ._backend import require_pristine_backend

    require_pristine_backend()
    from .expert_paging.pager import install_pager

    return install_pager(
        cache,
        geo,
        mode="ssd_llama",
        pack=Path(pack).expanduser().resolve(),
        host_rows=host_rows,
        **kwargs,
    )
