"""Frequency-recency and learned expert replacement policies."""

from collections import OrderedDict
import numpy as np


class History:
    def __init__(self, layers, experts):
        self.frequency = np.zeros((layers, experts), np.int64)
        self.last = np.full((layers, experts), -1, np.int64)
        self.clock = np.zeros(layers, np.int64)

    def observe(self, layer, ids):
        rows = np.asarray(ids).reshape(-1, np.asarray(ids).shape[-1])
        unique, count = np.unique(rows.reshape(-1), return_counts=True)
        self.frequency[layer, unique] += count
        ticks = np.repeat(np.arange(1, len(rows) + 1) + self.clock[layer], rows.shape[1])
        np.maximum.at(self.last[layer], rows.reshape(-1), ticks)
        self.clock[layer] += len(rows)

    def features(self, layer):
        last = self.last[layer]
        recency = np.where(last < 0, 0.0, 1.0 / (self.clock[layer] - last + 1))
        freq = self.frequency[layer].astype(np.float32)
        return np.concatenate((recency, freq / max(1.0, float(freq.max())))).astype(np.float32)


class GPUPlan:
    """Per-layer LRU/learned replacement, or global frequency-recency admission.

    SSD-LLaMA's unspecified combined score is lexicographic (frequency, recency).
    One whole-layer scratch reserve bounds cold prefill service without destroying
    the persistent hot cache. Every scratch byte is charged to the GPU budget.
    """

    def __init__(self, mode, layers, experts, slots, predictor=None):
        if mode not in {"ssd_llama", "flashmoe"}:
            raise ValueError(f"Unknown paper policy: {mode}")
        if mode == "flashmoe" and predictor is None:
            raise ValueError("FlashMoE requires a trained predictor")
        self.mode, self.layers, self.experts = mode, layers, experts
        self.slots, self.predictor = slots, predictor
        self.history = History(layers, experts)
        self.where = {}
        self.reverse = [-1] * slots
        self.lru = OrderedDict()
        self.tick = 0
        self.access = {}
        self.temp = experts if mode == "ssd_llama" else 0
        self.cap = slots // layers if not self.temp else slots - self.temp
        if self.temp and self.cap < experts:
            raise ValueError("SSD cache needs persistent and temporary capacity")
        if not self.temp and slots % layers:
            raise ValueError("Per-layer expert slots must divide the layer count")

    def score(self, expert):
        layer, local = divmod(expert, self.experts)
        return int(self.history.frequency[layer, local]), self.access.get(expert, -1)

    def plan(self, layer, local_ids):
        unique = list(dict.fromkeys(map(int, np.asarray(local_ids).reshape(-1))))
        active = {layer * self.experts + e for e in unique}
        if len(active) > (self.experts if self.temp else self.cap):
            raise ValueError("Active group exceeds cache capacity; group before dispatch")
        self.tick += 1
        for expert in active:
            self.access[expert] = self.tick
        predicted = None
        if self.mode == "flashmoe" and any(e not in self.where for e in active):
            if self.predictor is None:
                raise RuntimeError("FlashMoE requires an actually trained predictor")
            predicted = self.predictor(layer, self.history.features(layer))
        loads, mapping, temporary = [], {}, []
        for expert in [layer * self.experts + e for e in unique]:
            if expert in self.where:
                slot = self.where[expert]
                self.lru.move_to_end(expert)
            else:
                region = (
                    range(self.cap)
                    if self.temp
                    else range(layer * self.cap, (layer + 1) * self.cap)
                )
                slot = next((s for s in region if self.reverse[s] < 0), None)
                victim = None
                if slot is None:
                    candidates = [
                        e
                        for e in self.lru
                        if e not in active and (self.temp or e // self.experts == layer)
                    ]
                    if not candidates:
                        raise RuntimeError("All victim slots protected")
                    if self.temp:
                        victim = min(candidates, key=self.score)
                        if self.score(expert) <= self.score(victim):
                            slot = self.cap + len(temporary)
                            if slot >= self.slots:
                                raise RuntimeError("Temporary expert reserve exhausted")
                            temporary.append(expert)
                            victim = None
                        else:
                            slot = self.where[victim]
                    elif self.mode == "flashmoe":
                        victim = max(candidates, key=lambda e: float(predicted[e % self.experts]))
                        slot = self.where[victim]
                    else:
                        victim = candidates[0]
                        slot = self.where[victim]
                if victim is not None:
                    del self.where[victim]
                    del self.lru[victim]
                if slot < self.slots - self.temp:
                    self.where[expert] = slot
                    self.reverse[slot] = expert
                    self.lru[expert] = None
                loads.append((expert, slot, victim))
            mapping[expert % self.experts] = slot
        assert len(set(mapping.values())) == len(mapping)
        return mapping, loads, temporary
