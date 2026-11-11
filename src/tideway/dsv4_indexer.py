"""Bound the temporary memory of DSV4's causal indexer selection.

The score matrix is a fresh, exclusively consumed tensor in FreeToken's DSV4
Compressor/Indexer call sites. Preserve the same add-and-topk arithmetic while
avoiding the repeated int64 column matrix and a second full score allocation.
This opt-in module installs the Tideway low-memory indexer.
"""

_STATS = dict(enabled=False, calls=0, max_score_elements=0, max_mask_elements=0, tile_rows=1024)


def select_prefill(scores, *, start_pos, seqlen, ratio, topk, offset, tile_rows=1024):
    import torch

    if scores.shape[-2] != seqlen or scores.dtype != torch.float32 or scores.requires_grad:
        raise ValueError("Tideway DSV4 indexer requires exclusively owned inference FP32 scores")
    if tile_rows < 1:
        raise ValueError("tile_rows must be positive")
    n_blocks = scores.shape[-1]
    live = ((start_pos + torch.arange(1, seqlen + 1, device=scores.device)) // ratio).unsqueeze(1)
    columns = torch.arange(n_blocks, device=scores.device).unsqueeze(0)
    for begin in range(0, seqlen, tile_rows):
        end = min(seqlen, begin + tile_rows)
        # Addition (rather than replacement) preserves the baseline treatment
        # of infinities, signed zero and NaNs as well as ordinary finite scores.
        bias = torch.where(columns >= live[begin:end], float("-inf"), 0)
        scores[..., begin:end, :].add_(bias)
        del bias
    picks = scores.topk(min(topk, n_blocks), dim=-1)[1]
    return torch.where(picks >= live, -1, picks + offset)


def install():
    from freetoken.attention.dsv4_indexer import IndexerBackendMixin

    original = IndexerBackendMixin.indexer_select_prefill
    if getattr(original, "_tideway_lowmem", False):
        raise RuntimeError("Tideway DSV4 indexer already installed")

    def bounded(self, scores, **kwargs):
        _STATS["calls"] += 1
        _STATS["max_score_elements"] = max(_STATS["max_score_elements"], scores.numel())
        _STATS["max_mask_elements"] = max(
            _STATS["max_mask_elements"],
            min(kwargs["seqlen"], _STATS["tile_rows"]) * scores.shape[-1],
        )
        return select_prefill(scores, **kwargs, tile_rows=_STATS["tile_rows"])

    bounded._tideway_lowmem = True
    IndexerBackendMixin.indexer_select_prefill = bounded
    _STATS["enabled"] = True
    return original


def snapshot():
    return dict(_STATS)
