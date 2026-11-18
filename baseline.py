"""Synthetic MoE expert cache baseline.

This script generates a repeatable routing trace and replay it through NO_CACHE,
global LRU, and global LFU policies. Latency values are synthetic assumptions,
not measurements of inference or hardware.
"""

from __future__ import annotations

import csv
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


NUM_LAYERS = 6
EXPERTS_PER_LAYER = 32
ACTIVE_EXPERTS_PER_LAYER = 2
TOKEN_STEPS = 1200
SEED = 42
EXPERT_SIZE_MIB = 16
HOT_SET_INTERVAL = 250
TOTAL_MIB = (128, 256, 512)
MODELED_IO_LATENCY_MS_PER_MISS = 100.0

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class ExpertRequest:
    token_step: int
    layer: int
    expert: int
    hot_phase: int

    @property
    def key(self) -> str:
        return f"L{self.layer}:E{self.expert}"


def generate_trace(
    num_layers: int = NUM_LAYERS,
    experts_per_layer: int = EXPERTS_PER_LAYER,
    active_per_layer: int = ACTIVE_EXPERTS_PER_LAYER,
    token_steps: int = TOKEN_STEPS,
    seed: int = SEED,
    hot_set_interval: int = HOT_SET_INTERVAL,
) -> list[ExpertRequest]:
    """Generate a deterministic trace.

    Synthetic assumptions:
    * Each token routes to exactly two distinct experts in every layer.
    * The per-layer hot set contains 4 experts and is newly sampled every
      hot_set_interval tokens.
    * With seed 42, active experts are selected uniformly by random.sample:
      first from the current 4-expert hot set, and thus only those four are
      considered hot; no router probabilities or real model weights are used.
    """
    rng = random.Random(seed)
    requests: list[ExpertRequest] = []
    previous_hot_sets: dict[int, tuple[int, ...]] | None = None

    for step in range(token_steps):
        phase = step // hot_set_interval
        if step % hot_set_interval == 0:
            hot_sets: dict[int, tuple[int, ...]] = {}
            for layer in range(num_layers):
                pool = list(range(experts_per_layer))
                if previous_hot_sets is not None:
                    prior = set(previous_hot_sets[layer])
                    pool = [expert for expert in pool if expert not in prior]
                if len(pool) < 4:
                    pool = list(range(experts_per_layer))
                hot_sets[layer] = tuple(sorted(rng.sample(pool, 4)))
            previous_hot_sets = hot_sets

        assert previous_hot_sets is not None
        for layer in range(num_layers):
            active = rng.sample(previous_hot_sets[layer], active_per_layer)
            for expert in active:
                requests.append(ExpertRequest(step, layer, expert, phase))

    return requests


class NoCache:
    name = "NO_CACHE"

    def __init__(self, budget_mib: int, expert_size_mib: int = EXPERT_SIZE_MIB):
        del budget_mib
        self.expert_size_mib = expert_size_mib

    def request(self, key: str) -> bool:
        del key
        return False

    @property
    def used_mib(self) -> int:
        return 0


class LRUCache:
    name = "LRU"

    def __init__(self, budget_mib: int, expert_size_mib: int = EXPERT_SIZE_MIB):
        if budget_mib < 0:
            raise ValueError("budget_mib must not be negative")
        self.capacity_experts = budget_mib // expert_size_mib
        self.expert_size_mib = expert_size_mib
        self.entries: OrderedDict[str, None] = OrderedDict()

    def request(self, key: str) -> bool:
        if key in self.entries:
            self.entries.move_to_end(key)
            return True
        if len(self.entries) >= self.capacity_experts:
            self.entries.popitem(last=False)
        self.entries[key] = None
        return False

    @property
    def used_mib(self) -> int:
        return len(self.entries) * self.expert_size_mib


class LFUCache:
    name = "LFU"

    def __init__(self, budget_mib: int, expert_size_mib: int = EXPERT_SIZE_MIB):
        if budget_mib < 0:
            raise ValueError("budget_mib must not be negative")
        self.capacity_experts = budget_mib // expert_size_mib
        self.expert_size_mib = expert_size_mib
        self.entries: dict[str, int] = {}
        self.last_used: dict[str, int] = {}
        self.clock = 0

    def request(self, key: str) -> bool:
        self.clock += 1
        if key in self.entries:
            self.entries[key] += 1
            self.last_used[key] = self.clock
            return True
        if len(self.entries) >= self.capacity_experts:
            victim = min(
                self.entries,
                key=lambda item: (self.entries[item], self.last_used[item]),
            )
            del self.entries[victim]
            del self.last_used[victim]
        self.entries[key] = 1
        self.last_used[key] = self.clock
        return False

    @property
    def used_mib(self) -> int:
        return len(self.entries) * self.expert_size_mib


Cache = NoCache | LRUCache | LFUCache


def replay(
    requests: Iterable[ExpertRequest],
    cache: Cache,
) -> dict[str, int | float | str]:
    hits = 0
    misses = 0

    for request in requests:
        if cache.request(request.key):
            hits += 1
        else:
            misses += 1

    requests_count = hits + misses
    total_bytes_mib = misses * cache.expert_size_mib
    modeled_latency_ms = misses * MODELED_IO_LATENCY_MS_PER_MISS
    return {
        "policy": cache.name,
        "requests": requests_count,
        "hits": hits,
        "misses": misses,
        "miss_rate": misses / requests_count if requests_count else 0.0,
        "hit_rate": hits / requests_count if requests_count else 0.0,
        "total_expert_bytes_mib": total_bytes_mib,
        "modeled_io_latency_ms": modeled_latency_ms,
    }


def write_trace(trace: Sequence[ExpertRequest]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "synthetic_trace.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "token_step",
                "hot_phase",
                "layer_index",
                "expert_index",
                "expert_key",
            ]
        )
        for request in trace:
            writer.writerow(
                [
                    request.token_step,
                    request.hot_phase,
                    request.layer,
                    request.expert,
                    request.key,
                ]
            )
    return path


def write_results(rows: Sequence[dict[str, int | float | str]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "baseline.csv"
    fieldnames = [
        "cache_budget_mib",
        "policy",
        "requests",
        "hits",
        "misses",
        "miss_rate",
        "hit_rate",
        "total_expert_bytes_mib",
        "modeled_io_latency_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def print_table(rows: Sequence[dict[str, int | float | str]]) -> None:
    headers = [
        ("Budget MiB", "cache_budget_mib"),
        ("Policy", "policy"),
        ("Hits", "hits"),
        ("Misses", "misses"),
        ("Hit rate", "hit_rate"),
        ("Miss rate", "miss_rate"),
        ("Bytes MiB", "total_expert_bytes_mib"),
        ("I/O ms", "modeled_io_latency_ms"),
    ]
    print("  ".join(name for name, _ in headers))
    print("-" * 110)
    for row in rows:
        values = []
        for name, key in headers:
            value = row[key]
            if key in ("hit_rate", "miss_rate"):
                values.append(f"{float(value):.4f}")
            elif key == "modeled_io_latency_ms":
                values.append(f"{float(value):.1f}")
            else:
                values.append(str(value))
        print("  ".join(values))


def main() -> None:
    trace = generate_trace()
    write_trace(trace)

    rows: list[dict[str, int | float | str]] = []
    for budget in TOTAL_MIB:
        for cache_type in (NoCache, LRUCache, LFUCache):
            cache = cache_type(budget)
            result = replay(trace, cache)
            row: dict[str, int | float | str] = {
                "cache_budget_mib": budget,
                **result,
            }
            rows.append(row)

    write_results(rows)
    print_table(rows)
    print(
        "\nGenerated data/synthetic_trace.csv and "
        "results/baseline.csv. Trace requests:"
        f" {len(trace)}."
    )


if __name__ == "__main__":
    main()
