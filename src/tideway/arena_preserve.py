"""Preserve Decode experts outside a measured Prefill workspace bound.

Unknown or previously overflowing token buckets retain the native conservative
arena. A known successful bucket retains the native 25% + 8 row margin, without
the heuristic that rounds a >60%-full arena up to 100%. The C allocator and its
cudaMalloc fallback remain unchanged; a fallback disables shrinking that bucket
on subsequent calls. No forward, KV update, or allocation is retried.
"""

import math


class ArenaPreservation:
    def __init__(self, cache, shift_module):
        self.cache, self.module, self.owner = cache, shift_module, cache.shift
        if self.owner is None or not self.owner.arena_sized:
            raise ValueError("measured native Shift arena is required")
        self.original_estimate = shift_module._arena_rows_estimate
        self.original_after = shift_module.arena_after
        self.eligible = set(self.owner.arena_hw) if self.owner.arena_fallback_events == 0 else set()
        self.blocked = set()
        self.closed = False
        self.stats = dict(
            estimates=0,
            reduced=0,
            preserved_slot_estimates=0,
            unknown_buckets=0,
            observed_fallbacks=0,
            completed_prefills=0,
        )
        self.events = []

        def estimate(st, n_tok):
            if st is not self.owner:
                return self.original_estimate(st, n_tok)
            b = shift_module._arena_bucket(n_tok)
            native = self.original_estimate(st, n_tok)
            self.stats["estimates"] += 1
            if b in self.blocked:
                result = st.arena_main_rows
            elif b in self.eligible and b in st.arena_hw:
                result = max(1, min(math.ceil(st.arena_hw[b] * 1.25 + 8), st.arena_main_rows))
            else:
                self.stats["unknown_buckets"] += 1
                result = native
            if result < native:
                self.stats["reduced"] += 1
                self.stats["preserved_slot_estimates"] += native - result
            if len(self.events) < 256:
                self.events.append(
                    dict(
                        tokens=int(n_tok),
                        bucket=int(b),
                        native_main_rows=int(native),
                        selected_main_rows=int(result),
                        head_rows=int(st.arena_head),
                        saved_slots=int(native - result),
                        known_successful=b in self.eligible,
                        blocked_after_fallback=b in self.blocked,
                    )
                )
            return int(result)

        def after(st):
            if st is not self.owner:
                return self.original_after(st)
            before = st.arena_fallback_events
            result = self.original_after(st)
            b = shift_module._arena_bucket(st.arena_cur_tokens)
            self.stats["completed_prefills"] += 1
            if st.arena_fallback_events > before:
                self.blocked.add(b)
                self.eligible.discard(b)
                self.stats["observed_fallbacks"] += st.arena_fallback_events - before
            elif b not in self.blocked:
                self.eligible.add(b)
            return result

        self.estimate_hook, self.after_hook = estimate, after
        try:
            shift_module._arena_rows_estimate = estimate
            shift_module.arena_after = after
        except BaseException:
            if shift_module._arena_rows_estimate is estimate:
                shift_module._arena_rows_estimate = self.original_estimate
            if shift_module.arena_after is after:
                shift_module.arena_after = self.original_after
            raise

    def snapshot(self):
        return dict(
            enabled=True,
            policy="known-bucket measured margin; no 60%-to-full expansion",
            closed=self.closed,
            owner_unchanged=self.cache.shift is self.owner,
            counters=dict(self.stats),
            eligible_buckets=sorted(self.eligible),
            blocked_buckets=sorted(self.blocked),
            estimate_events=list(self.events),
            events_truncated=self.stats["estimates"] > len(self.events),
            changes_allocator=False,
            retries_forward=False,
            extra_expert_payload_bytes=0,
        )

    def close(self):
        if self.closed:
            return
        if (
            self.module._arena_rows_estimate is not self.estimate_hook
            or self.module.arena_after is not self.after_hook
        ):
            raise RuntimeError("Shift arena hooks changed while the controller was active")
        self.module._arena_rows_estimate = self.original_estimate
        self.module.arena_after = self.original_after
        self.closed = True


def install_arena_preservation(llm):
    from freetoken.moe import shift

    cache = llm.engine.moe_offload_cache
    if cache is None or cache.decode_target != "gpu" or "ensure_experts" in vars(cache):
        raise ValueError("arena preservation requires native GPU Tideway-Shift")
    return ArenaPreservation(cache, shift)
