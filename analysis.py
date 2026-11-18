"""Per-phase and seed-sweep analysis for the preserved synthetic baseline."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from baseline import (
    EXPERT_SIZE_MIB,
    LFUCache,
    LRUCache,
    NoCache,
    generate_trace,
)


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

PHASES = (
    (0, 249),
    (250, 499),
    (500, 749),
    (750, 999),
    (1000, 1199),
)
SEEDS = (42, 43, 44, 45, 46)
SWEEP_BUDGETS = (160, 192, 224, 256, 320, 384, 512)


def cache_types():
    return (NoCache, LRUCache, LFUCache)


def replay_detailed(trace, cache):
    hits = 0
    misses = 0
    evictions = 0
    cumulative_frequency: Counter[str] = Counter()
    phase_counts = {
        (start, end): {"hits": 0, "misses": 0, "evictions": 0}
        for start, end in PHASES
    }
    phase_snapshots = {}
    adaptation_counts = {
        start: {"hits": 0, "misses": 0}
        for start in (250, 500, 750, 1000)
    }

    for request in trace:
        key = request.key
        entries_before = set(
            getattr(cache, "entries", {}).keys()
            if hasattr(cache, "entries")
            else ()
        )
        hit = cache.request(key)
        cumulative_frequency[key] += 1

        phase = next(
            ((start, end) for start, end in PHASES
             if start <= request.token_step <= end),
            None,
        )
        if hit:
            hits += 1
            if phase is not None:
                phase_counts[phase]["hits"] += 1
            for start in adaptation_counts:
                if start <= request.token_step <= start + 49:
                    adaptation_counts[start]["hits"] += 1
        else:
            misses += 1
            if phase is not None:
                phase_counts[phase]["misses"] += 1
            for start in adaptation_counts:
                if start <= request.token_step <= start + 49:
                    adaptation_counts[start]["misses"] += 1

            if cache.name != "NO_CACHE" and len(entries_before) >= getattr(
                cache, "capacity_experts", 0
            ):
                evictions += 1
                if phase is not None:
                    phase_counts[phase]["evictions"] += 1

        if phase is not None and request.token_step == phase[1]:
            phase_snapshots[phase] = (
                cache.used_mib,
                resident_frequency_distribution(cache, cumulative_frequency),
            )

    phase_rows = []
    for start, end in PHASES:
        counts = phase_counts[(start, end)]
        requests = counts["hits"] + counts["misses"]
        transferred_mib = counts["misses"] * cache.expert_size_mib
        occupancy_mib, resident_frequency = phase_snapshots[(start, end)]
        phase_rows.append(
            {
                "phase": f"{start}-{end}",
                "start_token": start,
                "end_token": end,
                "requests": requests,
                "hits": counts["hits"],
                "misses": counts["misses"],
                "hit_rate": counts["hits"] / requests if requests else 0.0,
                "transferred_mib": transferred_mib,
                "cache_occupancy_mib": occupancy_mib,
                "evictions": counts["evictions"],
                "resident_cumulative_frequency_distribution": resident_frequency,
            }
        )

    adaptation_rows = []
    for start in (250, 500, 750, 1000):
        counts = adaptation_counts[start]
        requests = counts["hits"] + counts["misses"]
        adaptation_rows.append(
            {
                "start_token": start,
                "end_token": start + 49,
                "requests": requests,
                "hits": counts["hits"],
                "misses": counts["misses"],
                "hit_rate": counts["hits"] / requests if requests else 0.0,
            }
        )

    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "occupancy_mib": cache.used_mib,
        "phase_rows": phase_rows,
        "adaptation_rows": adaptation_rows,
    }


def resident_frequency_distribution(cache, cumulative_frequency):
    if cache.name == "NO_CACHE":
        return ""
    entries = getattr(cache, "entries", {})
    if cache.name == "LFU":
        resident = sorted(entries.keys())
    else:
        resident = sorted(entries.keys())
    histogram = Counter(cumulative_frequency[key] for key in resident)
    if not histogram:
        return ""
    return "|".join(
        f"{frequency}:{count}"
        for frequency, count in sorted(histogram.items())
    )


def write_phase_analysis(seed=42):
    trace = generate_trace(seed=seed)
    rows = []
    adaptation_rows = []
    for budget in (128, 256, 512):
        for cache_type in cache_types():
            cache = cache_type(budget)
            detailed = replay_detailed(trace, cache)
            for row in detailed["phase_rows"]:
                rows.append(
                    {
                        "seed": seed,
                        "cache_budget_mib": budget,
                        "policy": cache.name,
                        **row,
                    }
                )
            for row in detailed["adaptation_rows"]:
                adaptation_rows.append(
                    {
                        "seed": seed,
                        "cache_budget_mib": budget,
                        "policy": cache.name,
                        **row,
                    }
                )

    write_csv(
        RESULTS_DIR / "phase_analysis.csv",
        [
            "seed",
            "cache_budget_mib",
            "policy",
            "phase",
            "start_token",
            "end_token",
            "requests",
            "hits",
            "misses",
            "hit_rate",
            "transferred_mib",
            "cache_occupancy_mib",
            "evictions",
            "resident_cumulative_frequency_distribution",
        ],
        rows,
    )
    write_csv(
        RESULTS_DIR / "adaptation_50_tokens.csv",
        [
            "seed",
            "cache_budget_mib",
            "policy",
            "start_token",
            "end_token",
            "requests",
            "hits",
            "misses",
            "hit_rate",
        ],
        adaptation_rows,
    )
    return rows


def write_seed_summary():
    budgets = SWEEP_BUDGETS
    rows = []
    for seed in SEEDS:
        trace = generate_trace(seed=seed)
        for budget in budgets:
            for cache_type in cache_types():
                cache = cache_type(budget)
                detailed = replay_detailed(trace, cache)
                requests = detailed["hits"] + detailed["misses"]
                rows.append(
                    {
                        "seed": seed,
                        "cache_budget_mib": budget,
                        "policy": cache.name,
                        "requests": requests,
                        "hits": detailed["hits"],
                        "misses": detailed["misses"],
                        "hit_rate": detailed["hits"] / requests,
                        "miss_rate": detailed["misses"] / requests,
                        "transferred_mib": (
                            detailed["misses"] * cache.expert_size_mib
                        ),
                        "evictions": detailed["evictions"],
                        "cache_occupancy_mib": detailed["occupancy_mib"],
                        "modeled_io_latency_ms": (
                            detailed["misses"] * 100.0
                        ),
                    }
                )

    write_csv(
        RESULTS_DIR / "seed_summary.csv",
        [
            "seed",
            "cache_budget_mib",
            "policy",
            "requests",
            "hits",
            "misses",
            "hit_rate",
            "miss_rate",
            "transferred_mib",
            "evictions",
            "cache_occupancy_mib",
            "modeled_io_latency_ms",
        ],
        rows,
    )
    return rows


def alternate_order(trace):
    """Reorder each token's requests only; retain the same token expert set."""
    ordered = []
    for token in range(1200):
        token_requests = [request for request in trace if request.token_step == token]
        ordered.extend(reversed(token_requests))
    return ordered


def write_alternative_order(seed=42):
    original = generate_trace(seed=seed)
    alternate = alternate_order(original)
    rows = []
    for budget in (128, 256, 512):
        for cache_type in cache_types():
            cache = cache_type(budget)
            detailed = replay_detailed(alternate, cache)
            requests = detailed["hits"] + detailed["misses"]
            rows.append(
                {
                    "seed": seed,
                    "access_order": "reversed_requests_within_each_token",
                    "cache_budget_mib": budget,
                    "policy": cache.name,
                    "requests": requests,
                    "hits": detailed["hits"],
                    "misses": detailed["misses"],
                    "hit_rate": detailed["hits"] / requests,
                }
            )
    write_csv(
        RESULTS_DIR / "alternative_order.csv",
        [
            "seed",
            "access_order",
            "cache_budget_mib",
            "policy",
            "requests",
            "hits",
            "misses",
            "hit_rate",
        ],
        rows,
    )
    return rows


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(phase_rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    budgets = (128, 256, 512)
    policies = ("LRU", "LFU", "NO_CACHE")
    phases = [f"{start}-{end}" for start, end in PHASES]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for policy in policies:
        for budget in budgets:
            values = [
                next(
                    row["hit_rate"]
                    for row in phase_rows
                    if row["seed"] == 42
                    and row["cache_budget_mib"] == budget
                    and row["policy"] == policy
                    and row["phase"] == phase
                )
                for phase in phases
            ]
            axes[0].plot(phases, values, marker="o",
                         label=f"{policy} {budget} MiB")
    axes[0].set_title("Per-phase hit rate")
    axes[0].set_xlabel("Token phase")
    axes[0].set_ylabel("Hit rate")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    for policy in policies:
        for budget in budgets:
            values = [
                next(
                    row["transferred_mib"]
                    for row in phase_rows
                    if row["seed"] == 42
                    and row["cache_budget_mib"] == budget
                    and row["policy"] == policy
                    and row["phase"] == phase
                )
                for phase in phases
            ]
            axes[1].plot(phases, values, marker="o",
                         label=f"{policy} {budget} MiB")
    axes[1].set_title("Per-phase modeled transfer traffic")
    axes[1].set_xlabel("Token phase")
    axes[1].set_ylabel("Transferred MiB")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase_hit_rate_and_transfer.png", dpi=160)
    plt.close(fig)


def main():
    phase_rows = write_phase_analysis(seed=42)
    write_seed_summary()
    write_alternative_order(seed=42)
    make_figures(phase_rows)
    print("Wrote:")
    print("  results/phase_analysis.csv")
    print("  results/adaptation_50_tokens.csv")
    print("  results/seed_summary.csv")
    print("  results/alternative_order.csv")
    print("  figures/phase_hit_rate_and_transfer.png")


if __name__ == "__main__":
    main()
