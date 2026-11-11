"""Carry a still-owned pending GDN snapshot through chunk continuation.

Intermediate ChunkedReq objects are deliberately not committed to the radix
cache by FreeToken's overlap scheduler. Their ping-pong slots and next-write
index already cross the boundary; the pending snapshot length must accompany
them. A final extend of <=64 tokens creates no replacement snapshot, so losing
the length otherwise leaves only the later, finished decode state reusable.
No intermediate tree insertion, new buffer, copy, or cache eviction is added.
"""


class CarryStats(dict):
    def __init__(self):
        super().__init__(
            inherited_boundaries=[], short_final_chunk_inheritances=[], payload_checks=[]
        )
        self.pending_payloads = []

    def verify_pending(self, llm):
        if not self.pending_payloads:
            return
        import torch

        torch.cuda.synchronize()
        cache = llm.cache_manager.prefix_cache
        pool = llm.engine.linear_state_pool
        for boundary, slot, expected in self.pending_payloads:
            nodes = [
                n
                for n in cache._snapshot_nodes()
                if n.mamba_value == slot and cache._path_len(n) == boundary
            ]
            assert len(nodes) == 1, (
                "carried snapshot not retained at expected boundary",
                boundary,
                slot,
            )
            equal = {}
            for name, before in expected.items():
                after = (
                    getattr(pool, name)[:, slot]
                    .detach()
                    .to("cpu", memory_format=torch.contiguous_format)
                    .view(torch.uint8)
                )
                equal[name] = torch.equal(before, after)
            assert all(equal.values()), ("carried GDN payload changed", boundary, slot, equal)
            self["payload_checks"].append(
                dict(
                    boundary=boundary,
                    slot=slot,
                    all_bytes_equal=equal,
                    bytes=sum(t.numel() for t in expected.values()),
                )
            )
        self.pending_payloads.clear()


def install(verify_payload=False):
    from freetoken.scheduler.prefill import PrefillAdder

    original = PrefillAdder.try_add_one
    stats = CarryStats()

    def try_add(self, pending_req):
        previous = pending_req.chunked_req
        req = original(self, pending_req)
        if previous is not None and req is not None and req.mamba_ping_pong is not None:
            boundary = previous.mamba_last_track_seqlen
            if boundary is not None:
                assert 0 < boundary <= req.cached_len, (boundary, req.cached_len)
                assert req.mamba_ping_pong == previous.mamba_ping_pong
                assert req.mamba_next_track_idx == previous.mamba_next_track_idx
                req.mamba_last_track_seqlen = boundary
                stats["inherited_boundaries"].append(int(boundary))
                if req.extend_len <= 64:
                    stats["short_final_chunk_inheritances"].append(
                        dict(boundary=int(boundary), extend=int(req.extend_len))
                    )
                    if verify_payload:
                        import torch
                        from freetoken.core import get_global_ctx

                        torch.cuda.synchronize()
                        pool = get_global_ctx().linear_state_pool
                        slot = req.mamba_ping_pong[1 - req.mamba_next_track_idx]
                        before = {
                            name: getattr(pool, name)[:, slot]
                            .detach()
                            .to("cpu", memory_format=torch.contiguous_format)
                            .view(torch.uint8)
                            for name in ("conv_states", "recurrent_states")
                        }
                        stats.pending_payloads.append((int(boundary), int(slot), before))
        return req

    PrefillAdder.try_add_one = try_add
    return stats
