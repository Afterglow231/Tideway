"""Opt-in retention of reusable window state until native allocation pressure.

Installed on one Tideway CacheManager, not on backend classes. Finish-time and
elastic phase-boundary trims stop eagerly tombstoning already-resident windows.
The original matcher, dual-reference eviction and pool allocator remain intact.
Before a new window lock, split its oldest node at the page-aligned active-window
boundary: retaining a large node must not accidentally pin its entire payload.

No payload is copied or allocated here. Retention can reduce the VRAM returned
to expert slots by an elastic pool, so its performance/memory tradeoff must be
measured. This module is inactive until explicitly installed.
"""

_ABSENT = object()


def _restore(obj, name, previous, installed):
    if vars(obj).get(name, _ABSENT) is not installed:
        return False
    if previous is _ABSENT:
        delattr(obj, name)
    else:
        setattr(obj, name, previous)
    return True


class PrefixRetention:
    def __init__(self, cache_manager):
        cm = cache_manager
        if not getattr(cm, "is_swa", False) or not getattr(cm, "swa_paged", False):
            raise TypeError("prefix retention requires a paged window radix cache")
        if getattr(cm, "tideway_prefix_retention", None) is not None:
            raise RuntimeError("prefix retention is already installed")
        if cm.sliding_window_size <= 0 or cm.page_size <= 0:
            raise ValueError("invalid window/page geometry")
        self.cm = cm
        self.state = "installed"
        self.busy = 0
        self.tree = None
        self.tree_bindings = []
        self.bindings = []
        self.stats = dict(
            deferred_trim_calls=0,
            bounded_lock_calls=0,
            window_boundary_splits=0,
            pressure_calls=0,
            pressure_freed_tokens=0,
            pressure_shortfalls=0,
            tree_rebinds=0,
        )
        self.request = {key: 0 for key in self.stats}
        self.native_rebuild = cm.rebuild
        self.native_ensure = cm.ensure_swa_slots
        try:
            self._attach_tree(cm.prefix_cache)
            self._bind(cm, "rebuild", self._rebuild, self.bindings)
            self._bind(cm, "ensure_swa_slots", self._ensure_swa_slots, self.bindings)
            self._bind(cm, "tideway_prefix_retention", self, self.bindings)
        except BaseException:
            self._detach_tree()
            for obj, name, previous, installed in reversed(self.bindings):
                _restore(obj, name, previous, installed)
            raise

    @staticmethod
    def _bind(obj, name, installed, bindings):
        previous = vars(obj).get(name, _ABSENT)
        setattr(obj, name, installed)
        bindings.append((obj, name, previous, installed))

    def _count(self, key, n=1):
        self.stats[key] += n
        self.request[key] += n

    def _check_owner(self):
        if self.state != "installed":
            raise RuntimeError("prefix retention is " + self.state)
        if self.cm.prefix_cache is not self.tree:
            raise RuntimeError("prefix tree replaced outside CacheManager.rebuild")

    def _attach_tree(self, tree):
        for name in ("trim_head_swa", "inc_lock", "evict_swa", "match_prefix"):
            if not callable(getattr(tree, name, None)):
                raise TypeError("window radix cache is missing " + name)
        if (
            tree.page_size != self.cm.page_size
            or tree.sliding_window_size != self.cm.sliding_window_size
        ):
            raise RuntimeError("window radix and manager geometry differ")
        self.tree = tree
        self.native_trim = tree.trim_head_swa
        self.native_inc_lock = tree.inc_lock
        self._bind(tree, "trim_head_swa", self._defer_trim, self.tree_bindings)
        self._bind(tree, "inc_lock", self._bounded_inc_lock, self.tree_bindings)

    def _detach_tree(self):
        for obj, name, previous, installed in reversed(self.tree_bindings):
            _restore(obj, name, previous, installed)
        self.tree_bindings.clear()
        self.tree = None

    def _defer_trim(self, input_ids, keep_from):
        self._check_owner()
        if keep_from <= 0:
            return self.tree.empty
        self._count("deferred_trim_calls")
        # Deliberately retain no reference to input_ids. These are currently
        # allocated, evictable rows; native ensure_swa_slots can reclaim them.
        return self.tree.empty

    def _bounded_inc_lock(self, node):
        self._check_owner()
        self._count("bounded_lock_calls")
        page = int(self.cm.page_size)
        remaining = ((int(self.cm.sliding_window_size) + page - 1) // page) * page
        current = node
        while not current.is_root() and remaining > 0:
            if not current.swa_tombstone:
                if current.length > remaining:
                    # Native split_at preserves full/swa refs and moves any
                    # existing boundary UUID to the prefix side. Existing
                    # owners therefore still release their original window.
                    current.split_at(current.length - remaining)
                    self._count("window_boundary_splits")
                    break
                remaining -= current.length
            current = current.parent
        return self.native_inc_lock(node)

    def _ensure_swa_slots(self, n):
        self._check_owner()
        before = int(self.cm.swa_pool.swa_available_size())
        pressure = before < n
        if pressure:
            self._count("pressure_calls")
        self.busy += 1
        try:
            # Do not suppress eviction or broaden the allocation promise. The
            # original caller still handles an impossible request/locked pool.
            result = self.native_ensure(n)
            if pressure:
                after = int(self.cm.swa_pool.swa_available_size())
                self._count("pressure_freed_tokens", max(0, after - before))
                if after < n:
                    self._count("pressure_shortfalls")
            return result
        finally:
            self.busy -= 1

    def _rebuild(self, *args, **kwargs):
        self._check_owner()
        if self.busy:
            raise RuntimeError("cannot rebuild during window allocation")
        result = self.native_rebuild(*args, **kwargs)
        new_tree = self.cm.prefix_cache
        if new_tree is not self.tree:
            self._detach_tree()
            self._attach_tree(new_tree)
            self._count("tree_rebinds")
        return result

    def reset_stats(self):
        self.request = {key: 0 for key in self.stats}

    def snapshot(self):
        pool = self.cm.swa_pool
        return dict(
            enabled=self.state == "installed",
            state=self.state,
            mode="pressure_driven_window_retention",
            counters=dict(self.stats),
            request_counters=dict(self.request),
            window_tokens=int(self.cm.sliding_window_size),
            page_tokens=int(self.cm.page_size),
            pool_capacity_tokens=int(pool.swa_num_tokens) - 1,
            free_window_tokens=int(pool.swa_available_size()),
            evictable_window_tokens=int(self.cm.prefix_cache.swa_evictable_size),
            protected_window_tokens=int(self.cm.prefix_cache.swa_protected),
            extra_payload_bytes=0,
            input_ids_retained=False,
            matching="unchanged native window-valid prefix matcher",
            eviction="unchanged native pressure eviction; active locks preserved",
            elastic_tradeoff="retained windows may reduce expert slots within the same physical budget",
        )

    def close(self):
        if self.state == "closed":
            return
        if self.busy:
            raise RuntimeError("cannot close during window allocation")
        self._detach_tree()
        for obj, name, previous, installed in reversed(self.bindings):
            _restore(obj, name, previous, installed)
        self.bindings.clear()
        self.state = "closed"


def enable_prefix_retention(cache_manager):
    """Install on a Tideway-owned manager before common warmup; default is off."""
    return PrefixRetention(cache_manager)
