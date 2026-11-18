"""Adaptive cache policies used only by the Day 7 analysis."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class _FrequencyState:
    score: float
    last_decay_tick: int
    last_used_tick: int


class DecayedLFUCache:
    """LFU with lazy exponential aging and LRU tie-breaking.

    The logical clock advances once per request. Before an entry is updated
    or compared for eviction, its score is multiplied by
    ``alpha ** elapsed_requests``. The current request is then added to a hit
    score, or inserted with score 1 on a miss. Eviction compares only resident
    state at the current clock; no later requests are inspected.
    """

    name = "DECAYED_LFU"
    metadata_fields_per_entry = 3

    def __init__(self, capacity_experts: int, decay_alpha: float):
        if capacity_experts < 0:
            raise ValueError("capacity_experts must not be negative")
        if not 0.0 < decay_alpha <= 1.0:
            raise ValueError("decay_alpha must be in (0, 1]")
        self.capacity_experts = capacity_experts
        self.decay_alpha = decay_alpha
        self.entries: dict[str, _FrequencyState] = {}
        self.clock = 0
        self.evictions = 0

    def _decay(self, state: _FrequencyState, now: int) -> None:
        elapsed = now - state.last_decay_tick
        if elapsed:
            state.score *= self.decay_alpha ** elapsed
            state.last_decay_tick = now

    def frequency(self, key: str) -> float:
        state = self.entries[key]
        elapsed = self.clock - state.last_decay_tick
        return state.score * self.decay_alpha ** elapsed

    def request(self, key: str) -> bool:
        self.clock += 1
        state = self.entries.get(key)
        if state is not None:
            self._decay(state, self.clock)
            state.score += 1.0
            state.last_used_tick = self.clock
            return True

        if self.capacity_experts == 0:
            return False
        if len(self.entries) >= self.capacity_experts:
            for resident in self.entries.values():
                self._decay(resident, self.clock)
            victim = min(
                self.entries,
                key=lambda resident: (
                    self.entries[resident].score,
                    self.entries[resident].last_used_tick,
                ),
            )
            del self.entries[victim]
            self.evictions += 1

        self.entries[key] = _FrequencyState(
            score=1.0,
            last_decay_tick=self.clock,
            last_used_tick=self.clock,
        )
        return False

    @property
    def used_experts(self) -> int:
        return len(self.entries)


class SegmentedLRUCache:
    """Two-segment LRU with probationary admission and protected reuse.

    A new key enters the probationary LRU segment. A second request promotes
    it to the protected LRU segment. Protected overflow demotes its least
    recently used entry to probationary MRU; probationary overflow evicts its
    LRU entry. Capacity is split using ``ceil(capacity * probationary_ratio)``
    slots for probationary entries, with at least one probationary slot when
    capacity is nonzero.
    """

    name = "SLRU"
    metadata_fields_per_entry = 2

    def __init__(
        self,
        capacity_experts: int,
        probationary_ratio: float = 0.25,
    ):
        if capacity_experts < 0:
            raise ValueError("capacity_experts must not be negative")
        if not 0.0 < probationary_ratio < 1.0:
            raise ValueError("probationary_ratio must be in (0, 1)")
        self.capacity_experts = capacity_experts
        self.probationary_capacity = (
            max(1, math.ceil(capacity_experts * probationary_ratio))
            if capacity_experts else 0
        )
        self.protected_capacity = (
            capacity_experts - self.probationary_capacity
        )
        self.probationary: OrderedDict[str, None] = OrderedDict()
        self.protected: OrderedDict[str, None] = OrderedDict()
        self.evictions = 0

    def _evict_probationary_lru(self) -> None:
        if self.probationary:
            self.probationary.popitem(last=False)
            self.evictions += 1

    def _evict_protected_lru(self) -> None:
        if self.protected:
            self.protected.popitem(last=False)
            self.evictions += 1

    def request(self, key: str) -> bool:
        if key in self.protected:
            self.protected.move_to_end(key)
            return True

        if key in self.probationary:
            del self.probationary[key]
            if self.protected_capacity == 0:
                self.probationary[key] = None
                return True
            self.protected[key] = None
            if len(self.protected) > self.protected_capacity:
                demoted, _ = self.protected.popitem(last=False)
                self.probationary[demoted] = None
                if len(self.probationary) > self.probationary_capacity:
                    self._evict_probationary_lru()
            return True

        if self.capacity_experts == 0:
            return False
        if len(self.probationary) + len(self.protected) >= self.capacity_experts:
            if self.probationary:
                self._evict_probationary_lru()
            else:
                self._evict_protected_lru()
        self.probationary[key] = None
        if len(self.probationary) > self.probationary_capacity:
            self._evict_probationary_lru()
            self.probationary[key] = None
        return False

    @property
    def used_experts(self) -> int:
        return len(self.probationary) + len(self.protected)
