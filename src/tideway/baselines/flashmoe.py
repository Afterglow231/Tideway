"""FlashMoE: trained per-layer replacement with raw checkpoint paging."""

from pathlib import Path


def install(cache, *, geo, pack, predictor, host_rows, **kwargs):
    from ._backend import require_pristine_backend

    require_pristine_backend()
    if isinstance(predictor, (str, Path)):
        from .expert_paging.learned import Predictor

        predictor = Predictor(Path(predictor).expanduser().resolve())
    from .expert_paging.pager import install_pager

    kwargs.setdefault("raw_store", "checkpoint")
    return install_pager(
        cache,
        geo,
        mode="flashmoe",
        pack=Path(pack).expanduser().resolve(),
        predictor=predictor,
        host_rows=host_rows,
        **kwargs,
    )
