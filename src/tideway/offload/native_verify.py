"""Bounded-memory byte audit of a native full-host expert cache (diagnostic only)."""


def verify_native_cache(cache):
    import time
    import torch

    started = time.monotonic()
    torch.cuda.synchronize(cache.device)
    owners = cache.id_of_slot.detach().cpu().tolist()
    locations = cache.slot_for_id.detach().cpu().view(-1).tolist()
    checked = payload_bytes = 0
    for slot, expert in enumerate(owners):
        if expert < 0:
            continue
        if expert >= len(locations) or locations[expert] != slot:
            raise RuntimeError(
                f"non-bijective native expert mapping at slot={slot}, expert={expert}"
            )
        layer, local = divmod(expert, cache.num_experts)
        for name, gpu in cache.bank_caches.items():
            # One bank row at a time, so the audit never duplicates the host pool
            # or materializes the whole VRAM cache in RAM.
            actual = gpu[slot].detach().cpu().contiguous().view(torch.uint8)
            expected = cache.bank_sources[name][layer][local].contiguous().view(torch.uint8)
            if not torch.equal(actual, expected):
                raise RuntimeError(
                    f"native expert payload mismatch: slot={slot}, expert={expert}, bank={name}"
                )
            payload_bytes += actual.numel()
        checked += 1
    for expert, slot in enumerate(locations):
        if slot >= 0 and (slot >= len(owners) or owners[slot] != expert):
            raise RuntimeError(
                f"native expert reverse mapping mismatch: expert={expert}, slot={slot}"
            )
    return dict(
        ok=True,
        resident_rows=checked,
        payload_bytes=payload_bytes,
        seconds=time.monotonic() - started,
        host_working_set="one expert-bank row",
        diagnostic_only=True,
    )
