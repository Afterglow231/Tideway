"""Retain optional SWA history in Elastic's native rounding slack only.

Fixed physical pools keep the existing pressure-driven policy. Elastic pools
continue native trimming/compaction and may keep extra history only when the
current request's declared remaining generation bound fits the same rounded
SWA target. No future request, measured output length, or benchmark ID is read.
The native prefix matcher and existing bounded-window-lock adapter are reused.
"""

from .prefix_retention import PrefixRetention
import weakref


def dry_trim_details(tree, input_ids, keep_from):
    """Count native trim's freed rows without matching, splitting or stamping.

    A partial leaf would be split by native match_prefix first; its prefix then
    becomes internal and is eligible. A complete leaf is not trimmed. Unknown
    or non-CPU key geometry is rejected so callers can keep native behavior.
    """
    import torch

    page = int(tree.page_size)
    if type(keep_from) is not int or keep_from < 0 or keep_from % page:
        raise ValueError("trim boundary must be a nonnegative page multiple")
    if input_ids.device.type != "cpu" or input_ids.ndim != 1 or keep_from > len(input_ids):
        raise ValueError("dry trim requires current CPU token keys")
    node = tree.root
    pos = 0
    released = 0
    eligible_tail = 0
    while pos < keep_from:
        child = node.children.get(tree.key_fn(input_ids[pos:]))
        if child is None:
            raise ValueError("trim path is not fully represented")
        take = min(int(child.length), keep_from - pos)
        if (
            child._key.device.type != "cpu"
            or take % page
            or not torch.equal(child._key[:take], input_ids[pos : pos + take])
        ):
            raise ValueError("trim path has unmatched or unsupported keys")
        partial = take < child.length
        if (
            not child.swa_tombstone
            and child.swa_ref_count == 0
            and (partial or not child.is_leaf())
        ):
            released += take
            eligible_tail += take
        else:
            eligible_tail = 0
        pos += take
        if partial:
            break
        node = child
    return dict(releasable_rows=released, contiguous_releasable_tail_rows=eligible_tail)


def dry_trim_rows(tree, input_ids, keep_from):
    return dry_trim_details(tree, input_ids, keep_from)["releasable_rows"]


def declared_decode_guard(req, page, *, finished=False):
    """Current API budget, not actual future tokens; include extra/page guards."""
    from freetoken.scheduler.prefill import ChunkedReq

    if isinstance(req, ChunkedReq):
        return None  # An unfinished input chunk has no decode handoff yet.
    output = getattr(req, "output_len", None)
    declared = getattr(getattr(req, "sampling_params", None), "max_tokens", None)
    if (
        type(output) is not int
        or type(declared) is not int
        or output <= 0
        or declared <= 0
        or output > declared
    ):
        return None
    if finished:
        return dict(rows=page, source="finished_current_request", remaining_declared_tokens=0)
    maximum = getattr(req, "max_device_len", None)
    cached = getattr(req, "cached_len", None)
    if (
        any(type(v) is not int for v in (maximum, cached))
        or maximum < cached
        or maximum - output < 0
        or cached < maximum - output
    ):
        return None
    remaining = maximum - cached
    # Reserving the entire declared remainder is conservative even when native
    # periodic out-of-window freeing would make a tighter estimate possible.
    rows = ((remaining + 1 + page - 1) // page) * page + page
    return dict(
        rows=rows,
        source="current_request_declared_max_tokens",
        remaining_declared_tokens=remaining,
        declared_max_tokens=declared,
        scheduler_extra_guard_tokens=1,
        allocator_guard_pages=1,
    )


def plan_elastic_trim(tree, input_ids, keep_from, *, global_live, rounded, guard_rows):
    if (
        type(global_live) is not int
        or global_live < 0
        or type(guard_rows) is not int
        or guard_rows < 0
    ):
        raise ValueError("nonnegative live/guard row counts required")
    dry = dry_trim_details(tree, input_ids, keep_from)
    native_free = dry["releasable_rows"]
    if native_free > global_live or native_free > int(tree.swa_evictable_size):
        raise RuntimeError("native trim accounting exceeds live/evictable rows")
    native_live = global_live - native_free
    target = int(rounded(native_live + 1))  # +1 is the physical sentinel row.
    if target < native_live + 1:
        raise RuntimeError("native rounded target cannot cover its live rows")
    slack = max(0, target - 1 - native_live - guard_rows)
    page = int(tree.page_size)
    move = min(keep_from, dry["contiguous_releasable_tail_rows"], (slack // page) * page)
    chosen = keep_from - move
    chosen_free = dry_trim_rows(tree, input_ids, chosen)
    # Moving a frontier can turn a leaf into an internal node. Never let this
    # adapter trim more than native merely to claim a retention improvement.
    if chosen_free > native_free or native_free - chosen_free != move:
        chosen = keep_from
        chosen_free = native_free
    retained = native_free - chosen_free
    candidate_live = global_live - chosen_free
    if retained and candidate_live + guard_rows + 1 > target:
        raise RuntimeError("optional history exceeds native phase/decode capacity")
    return dict(
        native_keep_from=keep_from,
        selected_keep_from=chosen,
        global_live_rows=global_live,
        native_release_rows=native_free,
        selected_release_rows=chosen_free,
        native_minimum_live_rows=native_live,
        candidate_live_rows=candidate_live,
        native_rounded_swa_target=target,
        sentinel_rows=1,
        decode_guard_rows=guard_rows,
        available_rounding_slack=slack,
        optional_retained_rows=retained,
        candidate_plus_guard_and_sentinel=candidate_live + guard_rows + 1,
        guarantee=(
            "No optional rows added"
            if not retained
            else "Current phase and declared decode growth fit within native rounded SWA target"
        ),
    )


class CapacityAwarePrefixRetention(PrefixRetention):
    def __init__(self, cache_manager):
        self.elastic = getattr(cache_manager, "elastic_kv", None)
        self.active_guard = None
        self.pending_guards = {}
        self.trim_events = []
        self.capacity_stats = dict(
            elastic_trim_calls=0,
            optional_rows_retained=0,
            native_fallbacks=0,
            unknown_budget_fallbacks=0,
        )
        super().__init__(cache_manager)
        try:
            if self.elastic is not None:
                e = self.elastic
                self.native_cache_req_swa = cache_manager._cache_req_swa
                self.native_after_prefill = e.after_prefill

                def cache_req(req, *, finished):
                    old = self.active_guard
                    self.active_guard = declared_decode_guard(
                        req, int(cache_manager.page_size), finished=finished
                    )
                    try:
                        return self.native_cache_req_swa(req, finished=finished)
                    finally:
                        self.active_guard = old

                def after_prefill(cm, req):
                    self._check_owner()
                    if cm is not self.cm:
                        raise RuntimeError("foreign Elastic cache manager")
                    before = len(e._pending_swa_trim)
                    result = self.native_after_prefill(cm, req)
                    guard = declared_decode_guard(req, int(cm.page_size), finished=False)
                    # Only integer object IDs and current budget metadata remain;
                    # no new token/payload owner or future input is retained.
                    self.pending_guards = {
                        k: v for k, v in self.pending_guards.items() if v[0]() is not None
                    }
                    for ids, _ in e._pending_swa_trim[before:]:
                        self.pending_guards[id(ids)] = (weakref.ref(ids), guard)
                    return result

                self._bind(cache_manager, "_cache_req_swa", cache_req, self.bindings)
                self._bind(e, "after_prefill", after_prefill, self.bindings)
        except BaseException:
            self.close()
            raise

    def _elastic_eligible(self):
        e = self.elastic
        return (
            e is not None
            and getattr(self.cm, "elastic_kv", None) is e
            and e.cm is self.cm
            and e.pool is self.cm.swa_pool
            and not e._failed
            and e.compact_swa
            and "swa" in e.layouts
            and getattr(e.engine.config, "max_running_req", None) == 1
            and int(e.layouts["swa"].usable) == int(e.pool.swa_num_tokens)
        )

    def _defer_trim(self, input_ids, keep_from):
        if self.elastic is None:
            return super()._defer_trim(input_ids, keep_from)
        self._check_owner()
        self.capacity_stats["elastic_trim_calls"] += 1
        pending_entry = self.pending_guards.pop(id(input_ids), None)
        pending = (
            pending_entry[1]
            if pending_entry is not None and pending_entry[0]() is input_ids
            else None
        )
        guard = self.active_guard if self.active_guard is not None else pending
        if not self._elastic_eligible() or guard is None:
            self.capacity_stats["native_fallbacks"] += 1
            if guard is None:
                self.capacity_stats["unknown_budget_fallbacks"] += 1
            return self.native_trim(input_ids, keep_from)
        pool = self.cm.swa_pool
        live = int(pool.swa_num_tokens) - 1 - int(pool.swa_available_size())
        try:
            plan = plan_elastic_trim(
                self.tree,
                input_ids,
                int(keep_from),
                global_live=live,
                rounded=lambda n: self.elastic._rounded("swa", n),
                guard_rows=int(guard["rows"]),
            )
        except ValueError:
            self.capacity_stats["native_fallbacks"] += 1
            return self.native_trim(input_ids, keep_from)
        except BaseException:
            self.state = "failed"
            raise
        freed = self.native_trim(input_ids, plan["selected_keep_from"])
        if int(freed.numel()) != plan["selected_release_rows"]:
            self.state = "failed"
            raise RuntimeError("native trim differs from side-effect-free capacity plan")
        self.capacity_stats["optional_rows_retained"] += plan["optional_retained_rows"]
        self.trim_events.append(dict(**plan, guard_source=dict(guard)))
        self.trim_events = self.trim_events[-128:]
        return freed

    def _rebuild(self, *args, **kwargs):
        result = super()._rebuild(*args, **kwargs)
        self.pending_guards.clear()
        self.active_guard = None
        return result

    def snapshot(self):
        out = super().snapshot()
        out.update(
            mode="capacity_aware_prefix_retention",
            fixed_pool_policy="Original pressure-driven retention",
            elastic_policy="Native trim plus optional history fitting native rounded target after declared decode/page guards",
            capacity_counters=dict(self.capacity_stats),
            trim_events=list(self.trim_events),
            pending_budget_records=len(self.pending_guards),
            future_request_data_read=False,
            declaration="Current API max_tokens/output_len upper bounds only; no actual future output length",
            limitations=[
                "Single active Elastic request with original native compaction; unsupported/unknown budgets use native trim.",
                "Guarantee concerns the native rounded compaction target and current declared decode growth. Original allocator fragmentation/hysteresis is unchanged.",
                "This does not predict or reserve for the next request, and does not claim whole-workload speedups.",
            ],
        )
        return out

    def close(self):
        super().close()
        self.pending_guards.clear()
        self.active_guard = None


def install_capacity_aware_prefix_retention(cache_manager, engine):
    if getattr(engine, "cache_manager", cache_manager) is not cache_manager:
        raise RuntimeError("foreign engine cache manager")
    return CapacityAwarePrefixRetention(cache_manager)


enable_capacity_aware_prefix_retention = install_capacity_aware_prefix_retention
