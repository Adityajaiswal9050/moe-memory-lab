"""Day 3 audit: generator realism and separate routing workloads."""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from baseline import (
    EXPERT_SIZE_MIB,
    LFUCache,
    LRUCache,
    NoCache,
    generate_trace,
)
from day3_workloads import (
    abrupt_changes,
    gradual_drift,
    mixed_alternating_tasks,
    static_skewed,
)


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "day3"
FIGURES_DIR = ROOT / "figures" / "day3"
BUDGETS = (128, 256, 512)
SEEDS = (42, 43, 44, 45, 46)
PHASES = ((0, 249), (250, 499), (500, 749), (750, 999), (1000, 1199))


WORKLOADS = (
    ("static_skewed", static_skewed),
    ("abrupt_changes", abrupt_changes),
    ("gradual_drift", gradual_drift),
    ("mixed_alternating_tasks", mixed_alternating_tasks),
)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def replay(trace, cache):
    hits = misses = evictions = 0
    phase_counts = {
        phase: {"hits": 0, "misses": 0, "evictions": 0}
        for phase in PHASES
    }
    for request in trace:
        capacity = getattr(cache, "capacity_experts", 0)
        entries_before = len(getattr(cache, "entries", {}))
        hit = cache.request(request.key)
        phase = next(
            phase for phase in PHASES
            if phase[0] <= request.token_step <= phase[1]
        )
        if hit:
            hits += 1
            phase_counts[phase]["hits"] += 1
        else:
            misses += 1
            phase_counts[phase]["misses"] += 1
            if cache.name != "NO_CACHE" and entries_before >= capacity:
                evictions += 1
                phase_counts[phase]["evictions"] += 1

    rows = []
    for phase in PHASES:
        counts = phase_counts[phase]
        requests = counts["hits"] + counts["misses"]
        rows.append({
            "phase": f"{phase[0]}-{phase[1]}",
            "requests": requests,
            "hits": counts["hits"],
            "misses": counts["misses"],
            "hit_rate": counts["hits"] / requests if requests else 0.0,
            "transferred_mib": counts["misses"] * cache.expert_size_mib,
            "transferred_bytes": counts["misses"] * cache.expert_size_mib * 1024 * 1024,
            "evictions": counts["evictions"],
            "cache_occupancy_mib": cache.used_mib,
        })
    return hits, misses, evictions, rows


def audit_baseline():
    trace_rows = []
    histogram_rows = []
    per_phase_rows = []
    for seed in SEEDS:
        trace = generate_trace(seed=seed)
        overall = Counter(request.key for request in trace)
        per_layer = {}
        for layer in range(6):
            per_layer[layer] = sorted({
                request.expert for request in trace
                if request.layer == layer
            })

        phase_distinct = {}
        for start, end in PHASES:
            phase_requests = [
                request for request in trace
                if start <= request.token_step <= end
            ]
            phase_keys = sorted({request.key for request in phase_requests})
            phase_distinct[(start, end)] = len(phase_keys)
            per_layer_phase = {}
            for layer in range(6):
                per_layer_phase[layer] = len({
                    request.expert for request in phase_requests
                    if request.layer == layer
                })

            per_phase_rows.append({
                "seed": seed,
                "workload": "baseline_original",
                "phase": f"{start}-{end}",
                "distinct_expert_ids": len(phase_keys),
                "per_layer_distinct_mean": (
                    sum(per_layer_phase.values()) / 6
                ),
                "hot_set_requests": len(phase_requests),
                "cold_set_requests": 0,
                "hot_set_fraction": 1.0,
                "cold_set_fraction": 0.0,
            })

        histogram = Counter(overall.values())
        for frequency, expert_count in sorted(histogram.items()):
            histogram_rows.append({
                "seed": seed,
                "workload": "baseline_original",
                "frequency": frequency,
                "expert_count": expert_count,
            })

        trace_rows.append({
            "seed": seed,
            "workload": "baseline_original",
            "overall_distinct_expert_ids": len(overall),
            "per_layer_distinct_ids": "|".join(
                f"L{layer}:{len(per_layer[layer])}"
                for layer in sorted(per_layer)
            ),
            "hot_set_request_fraction": 1.0,
            "cold_set_request_fraction": 0.0,
            "requests": len(trace),
        })

    write_csv(
        RESULTS_DIR / "baseline_audit_summary.csv",
        [
            "seed", "workload", "overall_distinct_expert_ids",
            "per_layer_distinct_ids", "hot_set_request_fraction",
            "cold_set_request_fraction", "requests",
        ],
        trace_rows,
    )
    write_csv(
        RESULTS_DIR / "baseline_audit_phase.csv",
        [
            "seed", "workload", "phase", "distinct_expert_ids",
            "per_layer_distinct_mean", "hot_set_requests",
            "cold_set_requests", "hot_set_fraction", "cold_set_fraction",
        ],
        per_phase_rows,
    )
    write_csv(
        RESULTS_DIR / "baseline_audit_histogram.csv",
        ["seed", "workload", "frequency", "expert_count"],
        histogram_rows,
    )


def hot_sets_for_workload(workload_name, seed):
    """Reconstruct labeled hot sets for cold/hot accounting."""
    rng = random.Random(seed)
    if workload_name == "static_skewed":
        hot_sets = {
            layer: tuple(sorted(rng.sample(range(32), 4)))
            for layer in range(6)
        }
        hot_by_step = {}
        for step in range(1200):
            hot_by_step[step] = hot_sets
            for layer in range(6):
                hot_set = hot_sets[layer]
                cold_set = tuple(
                    expert for expert in range(32)
                    if expert not in hot_set
                )
                chosen = []
                for _ in range(2):
                    pool = hot_set if rng.random() < 0.90 else cold_set
                    choices = [expert for expert in pool if expert not in chosen]
                    if not choices:
                        choices = [expert for expert in range(32)
                                   if expert not in chosen]
                    chosen.append(rng.choice(choices))
        return hot_by_step
    if workload_name == "abrupt_changes":
        hot_by_step = {}
        previous = {}
        for step in range(1200):
            if step % 250 == 0:
                hot_sets = {}
                for layer in range(6):
                    pool = list(range(32))
                    if layer in previous:
                        prior = set(previous[layer])
                        pool = [expert for expert in pool if expert not in prior]
                        if len(pool) < 4:
                            pool = list(range(32))
                    hot_sets[layer] = tuple(sorted(rng.sample(pool, 4)))
                previous = hot_sets
            hot_by_step[step] = previous
            for layer in range(6):
                hot_set = previous[layer]
                cold_set = tuple(
                    expert for expert in range(32)
                    if expert not in hot_set
                )
                chosen = []
                for _ in range(2):
                    pool = hot_set if rng.random() < 0.90 else cold_set
                    choices = [expert for expert in pool if expert not in chosen]
                    if not choices:
                        choices = [expert for expert in range(32)
                                   if expert not in chosen]
                    chosen.append(rng.choice(choices))
        return hot_by_step
    if workload_name == "mixed_alternating_tasks":
        task_sets = []
        for _ in (0, 1):
            task_sets.append({
                layer: tuple(sorted(rng.sample(range(32), 8)))
                for layer in range(6)
            })
        hot_by_step = {}
        for step in range(1200):
            hot_sets = task_sets[(step // 100) % 2]
            hot_by_step[step] = hot_sets
            for layer in range(6):
                hot_set = hot_sets[layer]
                cold_set = tuple(
                    expert for expert in range(32)
                    if expert not in hot_set
                )
                chosen = []
                for _ in range(2):
                    pool = hot_set if rng.random() < 0.85 else cold_set
                    choices = [expert for expert in pool if expert not in chosen]
                    if not choices:
                        choices = [expert for expert in range(32)
                                   if expert not in chosen]
                    chosen.append(rng.choice(choices))
        return hot_by_step
    gradual = {}
    for step in range(1200):
        step_hot_sets = {}
        center_float = (step / 1199) * 31
        for layer in range(6):
            center = (center_float + layer * 3) % 32
            weighted = []
            for expert in range(32):
                distance = (expert - center) % 32
                distance = min(distance, 32 - distance)
                weighted.append((expert, (32 - distance) ** 2))
            weighted.sort(key=lambda item: item[1], reverse=True)
            step_hot_sets[layer] = tuple(
                sorted(expert for expert, _ in weighted[:8])
            )
        gradual[step] = step_hot_sets
    return gradual


def workload_audit_and_sweep():
    summary_rows = []
    phase_rows = []
    histogram_rows = []
    seed_phase_rows = []
    for workload_name, generator in WORKLOADS:
        for seed in SEEDS:
            trace = generator(seed=seed)
            hot_by_step = hot_sets_for_workload(workload_name, seed)
            hot_requests = 0
            cold_requests = 0
            overall = Counter(request.key for request in trace)
            for request in trace:
                hot_set = hot_by_step[request.token_step][request.layer]
                if request.expert in hot_set:
                    hot_requests += 1
                else:
                    cold_requests += 1

            summary_rows.append({
                "seed": seed,
                "workload": workload_name,
                "overall_distinct_expert_ids": len(overall),
                "requests": len(trace),
                "hot_set_requests": hot_requests,
                "cold_set_requests": cold_requests,
                "hot_set_fraction": hot_requests / len(trace),
                "cold_set_fraction": cold_requests / len(trace),
            })
            histogram = Counter(overall.values())
            for frequency, expert_count in sorted(histogram.items()):
                histogram_rows.append({
                    "seed": seed,
                    "workload": workload_name,
                    "frequency": frequency,
                    "expert_count": expert_count,
                })

            for start, end in PHASES:
                phase_requests = [
                    request for request in trace
                    if start <= request.token_step <= end
                ]
                hot = cold = 0
                for request in phase_requests:
                    hot_set = hot_by_step[request.token_step][request.layer]
                    if request.expert in hot_set:
                        hot += 1
                    else:
                        cold += 1
                phase_rows.append({
                    "seed": seed,
                    "workload": workload_name,
                    "phase": f"{start}-{end}",
                    "distinct_expert_ids": len({
                        request.key for request in phase_requests
                    }),
                    "hot_set_requests": hot,
                    "cold_set_requests": cold,
                    "hot_set_fraction": hot / len(phase_requests),
                    "cold_set_fraction": cold / len(phase_requests),
                })

            for budget in BUDGETS:
                for cache_type in (NoCache, LRUCache, LFUCache):
                    cache = cache_type(budget)
                    hits, misses, evictions, cache_phase_rows = replay(
                        trace, cache
                    )
                    seed_phase_rows.append({
                        "seed": seed,
                        "workload": workload_name,
                        "cache_budget_mib": budget,
                        "policy": cache.name,
                        "requests": hits + misses,
                        "hits": hits,
                        "misses": misses,
                        "hit_rate": hits / (hits + misses),
                        "transferred_mib": misses * EXPERT_SIZE_MIB,
                        "transferred_bytes": (
                            misses * EXPERT_SIZE_MIB * 1024 * 1024
                        ),
                        "evictions": evictions,
                    })
                    for row in cache_phase_rows:
                        seed_phase_rows[-1].update({})  # no-op: summary remains separate

                    write_workload_phase_rows(
                        workload_name,
                        seed,
                        budget,
                        cache.name,
                        cache_phase_rows,
                        append=True,
                    )

    write_csv(
        RESULTS_DIR / "workload_summary.csv",
        [
            "seed", "workload", "overall_distinct_expert_ids", "requests",
            "hot_set_requests", "cold_set_requests", "hot_set_fraction",
            "cold_set_fraction",
        ],
        summary_rows,
    )
    write_csv(
        RESULTS_DIR / "workload_phase_fractions.csv",
        [
            "seed", "workload", "phase", "distinct_expert_ids",
            "hot_set_requests", "cold_set_requests", "hot_set_fraction",
            "cold_set_fraction",
        ],
        phase_rows,
    )
    write_csv(
        RESULTS_DIR / "workload_frequency_histogram.csv",
        ["seed", "workload", "frequency", "expert_count"],
        histogram_rows,
    )
    write_csv(
        RESULTS_DIR / "workload_seed_summary.csv",
        [
            "seed", "workload", "cache_budget_mib", "policy", "requests",
            "hits", "misses", "hit_rate", "transferred_mib",
            "transferred_bytes", "evictions",
        ],
        seed_phase_rows,
    )
    make_variance_summary(seed_phase_rows)
    make_figures(seed_phase_rows)


WORKLOAD_PHASE_PATH = RESULTS_DIR / "workload_phase_cache.csv"


def write_workload_phase_rows(
    workload_name, seed, budget, policy, rows, append=False
):
    fieldnames = [
        "seed", "workload", "cache_budget_mib", "policy", "phase",
        "requests", "hits", "misses", "hit_rate", "transferred_mib",
        "transferred_bytes", "evictions", "cache_occupancy_mib",
    ]
    output_rows = []
    if append and WORKLOAD_PHASE_PATH.exists():
        with WORKLOAD_PHASE_PATH.open("r", newline="", encoding="utf-8") as handle:
            output_rows.extend(csv.DictReader(handle))
    for row in rows:
        output_rows.append({
            "seed": seed,
            "workload": workload_name,
            "cache_budget_mib": budget,
            "policy": policy,
            **row,
        })
    write_csv(WORKLOAD_PHASE_PATH, fieldnames, output_rows)


def make_variance_summary(seed_rows):
    grouped = {}
    for row in seed_rows:
        key = (row["workload"], row["cache_budget_mib"], row["policy"])
        grouped.setdefault(key, []).append(row)
    rows = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2])):
        values = [row["hit_rate"] for row in grouped[key]]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        rows.append({
            "workload": key[0],
            "cache_budget_mib": key[1],
            "policy": key[2],
            "seed_count": len(values),
            "mean_hit_rate": mean,
            "hit_rate_variance": variance,
            "hit_rate_min": min(values),
            "hit_rate_max": max(values),
        })
    write_csv(
        RESULTS_DIR / "repeated_seed_variance.csv",
        [
            "workload", "cache_budget_mib", "policy", "seed_count",
            "mean_hit_rate", "hit_rate_variance", "hit_rate_min",
            "hit_rate_max",
        ],
        rows,
    )


def make_figures(seed_rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for workload_name, _ in WORKLOADS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for budget in BUDGETS:
            for policy in ("LRU", "LFU", "NO_CACHE"):
                values = [
                    row["hit_rate"] for row in seed_rows
                    if row["workload"] == workload_name
                    and row["cache_budget_mib"] == budget
                    and row["policy"] == policy
                    and row["seed"] == 42
                ]
                if values:
                    axes[0].plot(
                        [0], values, marker="o",
                        label=f"{policy} {budget} MiB",
                    )
        axes[0].set_title(f"{workload_name}: seed 42 aggregate hit rate")
        axes[0].set_ylabel("Hit rate")
        axes[0].set_ylim(0, 1)
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=7)

        for budget in BUDGETS:
            means = []
            for policy in ("LRU", "LFU", "NO_CACHE"):
                rows = [
                    row for row in seed_rows
                    if row["workload"] == workload_name
                    and row["cache_budget_mib"] == budget
                    and row["policy"] == policy
                ]
                means.append(
                    sum(row["transferred_mib"] for row in rows) / len(rows)
                    if rows else 0
                )
            axes[1].bar(
                [f"{budget} MiB\nLRU", f"{budget} MiB\nLFU",
                 f"{budget} MiB\nNO_CACHE"],
                means,
                alpha=0.8,
            )
        axes[1].set_title("Repeated-seed mean modeled transfer")
        axes[1].set_ylabel("Transferred MiB")
        axes[1].grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR / f"{workload_name}_hit_and_transfer.png",
            dpi=160,
        )
        plt.close(fig)


def main():
    audit_baseline()
    workload_audit_and_sweep()
    print("Wrote Day 3 audit and benchmark outputs under:")
    print("  results/day3/")
    print("  figures/day3/")


if __name__ == "__main__":
    main()
