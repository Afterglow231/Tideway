"""Atomic expert-slot LRU with stable physical-slot tie breaking."""

import numpy as np


class Cache:
    def __init__(self, capacity, universe, initial=None):
        if capacity < 1 or universe < 1:
            raise ValueError("positive capacity and universe required")
        self.slots = np.full(capacity, -1, dtype=np.int64)
        self.usage = np.zeros(capacity, dtype=np.int64)
        self.where = np.full(universe, -1, dtype=np.int64)
        self.step = 0
        if initial is not None:
            if len(initial.slots) != capacity:
                raise ValueError("initial state capacity mismatch")
            for k in ("slots", "usage", "where"):
                setattr(self, k, getattr(initial, k).copy())
            self.step = initial.step

    def access(self, ids):
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        if not ids.size:
            return 0
        if len(ids) > len(self.slots):
            raise ValueError("capacity cannot hold one atomic layer call")
        if ids.min() < 0 or ids.max() >= len(self.where):
            raise ValueError("expert outside universe")
        self.step += 1
        positions = self.where[ids]
        hits = positions[positions >= 0]
        self.usage[hits] = self.step
        missing = ids[positions < 0]  # sorted IDs, matches flashlib miss ranking
        available = np.flatnonzero(self.usage != self.step)
        order = np.argsort(self.usage[available], kind="stable")
        victims = available[order[: len(missing)]]
        old = self.slots[victims]
        self.last_evicted = old[old >= 0].copy()
        self.where[old[old >= 0]] = -1
        self.slots[victims] = missing
        self.where[missing] = victims
        self.usage[victims] = self.step
        assert np.all(self.where[ids] >= 0)
        return len(missing)
