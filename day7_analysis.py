"""Synthetic-only decay selection and exploratory Day 6 cache comparisons."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from baseline import (
    EXPERT_SIZE_MIB,
    LFUCache,
    LRUCache,
    NoCache,
)
from day3_workloads import (
    abrupt_changes,
    gradual_drift,
    mixed_alternating_tasks,
    static_skewed,
)
from day5_analysis import (
    CACHE_BUDGETS_MIB,
    MIB_BYTES,
    expert_storage_from_safetensors,
    load_config,
    write_csv,
)
from day6_analysis import (
    CATEGORY_ORDER,
    read_trace as read_day6_trace,
    validate_day6_trace,
)
from day7_policies import DecayedLFUCache, SegmentedLRUCache


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "day7"
FIGURES_DIR = ROOT / "figures" / "day7"
DAY6_REPORT_PATH = ROOT / "results" / "day6" / "pilot_report.json"
DAY6_EXPERTS_PATH = ROOT / "real_trace" / "model_source" / "model.safetensors"
SEEDS = (42, 43, 44, 45, 46)
DECAY_ALPHAS = (0.90, 0.95, 0.98, 0.99, 0.995, 0.999)
SYNTHETIC_BUDGETS_MIB = CACHE_BUDGETS_MIB
SYNTHETIC_PHASES = (
    (0, 249),
    (250, 499),
    (500, 749),
    (750, 999),
    (1000, 1199),
)
WORKLOADS = (
    ("static_skewed", static_skewed),
    ("abrupt_changes", abrupt_changes),
    ("gradual_drift", gradual_drift),
    ("mixed_alternating_tasks", mixed_alternating_tasks),
)
POLICY_ORDER = ("NO_CACHE", "LRU", "LFU", "DECAYED_LFU", "SLRU")
METADATA_FIELDS = {
    "NO_CACHE": 0,
    "LRU": 1,
    "LFU": 2,
    "DECAYED_LFU": 3,
    "SLRU": 2,
}
GLOBAL_METADATA_FIELDS = {
    "NO_CACHE": 0,
    "LRU": 0,
    "LFU": 1,
    "DECAYED_LFU": 1,
    "SLRU": 0,
}
METADATA_FIELD_BYTES = 8
SLRU_PROBATIONARY_RATIO = 0.25


def _cache(policy: str, capacity: int, alpha: float | None = None):
    if policy == "NO_CACHE":
        return NoCache(budget_mib=0, expert_size_mib=1)
    if policy == "LRU":
        return LRUCache(budget_mib=capacity, expert_size_mib=1)
    if policy == "LFU":
        return LFUCache(budget_mib=capacity, expert_size_mib=1)
    if policy == "DECAYED_LFU":
        if alpha is None:
            raise ValueError("DECAYED_LFU requires a decay coefficient")
        return DecayedLFUCache(capacity, alpha)
    if policy == "SLRU":
        return SegmentedLRUCache(
            capacity,
            probationary_ratio=SLRU_PROBATIONARY_RATIO,
        )
    raise ValueError(f"Unknown policy: {policy}")


def _entry_count(cache) -> int:
    if isinstance(cache, SegmentedLRUCache):
        return cache.used_experts
    return len(getattr(cache, "entries", {}))


def _policy_capacity(cache) -> int:
    return int(getattr(cache, "capacity_experts", 0))


def _estimated_state(cache, policy: str, resident_count: int) -> dict:
    resident_bytes = (
        resident_count
        * METADATA_FIELDS[policy]
        * METADATA_FIELD_BYTES
    )
    global_bytes = GLOBAL_METADATA_FIELDS[policy] * METADATA_FIELD_BYTES
    return {
        "resident_policy_metadata_fields_per_entry": METADATA_FIELDS[policy],
        "estimated_resident_metadata_bytes": resident_bytes,
        "estimated_global_metadata_bytes": global_bytes,
        "estimated_policy_metadata_bytes": resident_bytes + global_bytes,
        "state_overhead_model": (
            "Fixed-width 8-byte policy fields; excludes keys, containers, "
            "allocator, and Python object overhead."
        ),
    }


def _phase_summary_rows(
    counts_by_phase: dict,
    phase_occupancy: dict,
    phase_peak: dict,
    cache,
    policy: str,
    expert_bytes: int,
    budget_mib: int,
    alpha: float | None,
    scenario: str,
    workload: str,
    seed: int | str,
    prompt_category: str,
    warmup_requests: int,
) -> list[dict]:
    capacity = _policy_capacity(cache)
    rows = []
    for phase in list(counts_by_phase):
        counts = counts_by_phase[phase]
        requests = counts["hits"] + counts["misses"]
        end_count = phase_occupancy.get(phase, _entry_count(cache))
        state = _estimated_state(cache, policy, end_count)
        rows.append({
            "scenario": scenario,
            "workload": workload,
            "seed": seed,
            "prompt_category": prompt_category,
            "cache_budget_mib": budget_mib,
            "policy": policy,
            "decay_alpha": alpha if policy == "DECAYED_LFU" else "",
            "phase": phase,
            "requests": requests,
            "hits": counts["hits"],
            "misses": counts["misses"],
            "hit_rate": counts["hits"] / requests if requests else 0.0,
            "miss_rate": counts["misses"] / requests if requests else 0.0,
            "modeled_expert_bytes_transferred": counts["misses"] * expert_bytes,
            "evictions": counts["evictions"],
            "cache_capacity_experts": capacity,
            "cache_occupancy_bytes_at_boundary": end_count * expert_bytes,
            "peak_resident_experts": phase_peak.get(phase, end_count),
            **state,
            "warmup_requests_excluded": warmup_requests,
            "traffic_is_modeled_not_measured": True,
        })

    all_counts = {
        name: sum(values[name] for values in counts_by_phase.values())
        for name in ("hits", "misses", "evictions")
    }
    all_requests = all_counts["hits"] + all_counts["misses"]
    end_count = _entry_count(cache)
    rows.append({
        "scenario": scenario,
        "workload": workload,
        "seed": seed,
        "prompt_category": prompt_category,
        "cache_budget_mib": budget_mib,
        "policy": policy,
        "decay_alpha": alpha if policy == "DECAYED_LFU" else "",
        "phase": "all",
        "requests": all_requests,
        "hits": all_counts["hits"],
        "misses": all_counts["misses"],
        "hit_rate": all_counts["hits"] / all_requests if all_requests else 0.0,
        "miss_rate": all_counts["misses"] / all_requests if all_requests else 0.0,
        "modeled_expert_bytes_transferred": all_counts["misses"] * expert_bytes,
        "evictions": all_counts["evictions"],
        "cache_capacity_experts": capacity,
        "cache_occupancy_bytes_at_boundary": end_count * expert_bytes,
        "peak_resident_experts": max(phase_peak.values(), default=end_count),
        **_estimated_state(cache, policy, end_count),
        "warmup_requests_excluded": warmup_requests,
        "traffic_is_modeled_not_measured": True,
    })
    return rows


def replay_sequence(
    events: list[dict],
    cache,
    policy: str,
    expert_bytes: int,
    budget_mib: int,
    alpha: float | None,
    scenario: str,
    workload: str,
    seed: int | str,
    prompt_category: str,
    warmup_requests: int = 0,
) -> tuple[list[dict], list[dict]]:
    counts_by_phase = defaultdict(
        lambda: {"hits": 0, "misses": 0, "evictions": 0}
    )
    layer_counts = defaultdict(
        lambda: {"hits": 0, "misses": 0, "evictions": 0}
    )
    phase_occupancy = {}
    phase_peak = defaultdict(int)
    per_layer_peak = defaultdict(int)
    layers = sorted({event["layer_index"] for event in events})

    for position, event in enumerate(events):
        phase = event["phase"]
        layer = event["layer_index"]
        before_entries = _entry_count(cache)
        before_evictions = getattr(cache, "evictions", None)
        hit = cache.request(event["key"])
        if hit:
            counts_by_phase[phase]["hits"] += 1
            layer_counts[(phase, layer)]["hits"] += 1
        else:
            counts_by_phase[phase]["misses"] += 1
            layer_counts[(phase, layer)]["misses"] += 1

        after_evictions = getattr(cache, "evictions", None)
        if before_evictions is not None and after_evictions is not None:
            evicted = after_evictions - before_evictions
        else:
            evicted = int(
                not hit
                and policy != "NO_CACHE"
                and before_entries >= _policy_capacity(cache)
            )
        counts_by_phase[phase]["evictions"] += evicted
        layer_counts[(phase, layer)]["evictions"] += evicted

        resident_count = _entry_count(cache)
        phase_peak[phase] = max(phase_peak[phase], resident_count)
        per_layer_peak[(phase, layer)] = max(
            per_layer_peak[(phase, layer)],
            resident_count,
        )
        next_phase = (
            events[position + 1]["phase"]
            if position + 1 < len(events) else None
        )
        if next_phase != phase:
            phase_occupancy[phase] = resident_count

    summary_rows = _phase_summary_rows(
        counts_by_phase,
        phase_occupancy,
        phase_peak,
        cache,
        policy,
        expert_bytes,
        budget_mib,
        alpha,
        scenario,
        workload,
        seed,
        prompt_category,
        warmup_requests,
    )
    layer_rows = []
    for phase in (*counts_by_phase.keys(), "all"):
        for layer in layers:
            if phase == "all":
                counts = {
                    field: sum(
                        layer_counts[(phase_name, layer)][field]
                        for phase_name in counts_by_phase
                    )
                    for field in ("hits", "misses", "evictions")
                }
            else:
                counts = layer_counts[(phase, layer)]
            requests = counts["hits"] + counts["misses"]
            layer_rows.append({
                "scenario": scenario,
                "workload": workload,
                "seed": seed,
                "prompt_category": prompt_category,
                "cache_budget_mib": budget_mib,
                "policy": policy,
                "decay_alpha": alpha if policy == "DECAYED_LFU" else "",
                "phase": phase,
                "layer_index": layer,
                "requests": requests,
                "hits": counts["hits"],
                "misses": counts["misses"],
                "hit_rate": counts["hits"] / requests if requests else 0.0,
                "miss_rate": counts["misses"] / requests if requests else 0.0,
                "modeled_expert_bytes_transferred": (
                    counts["misses"] * expert_bytes
                ),
                "evictions": counts["evictions"],
                "cache_occupancy_bytes_at_boundary": (
                    phase_occupancy.get(phase, _entry_count(cache))
                    * expert_bytes
                ),
                "peak_resident_experts": per_layer_peak.get(
                    (phase, layer),
                    _entry_count(cache),
                ),
                **_estimated_state(
                    cache,
                    policy,
                    _entry_count(cache),
                ),
                "traffic_is_modeled_not_measured": True,
            })
    return summary_rows, layer_rows


def _synthetic_events(trace) -> list[dict]:
    events = []
    for order, request in enumerate(trace):
        phase = next(
            f"{start}-{end}"
            for start, end in SYNTHETIC_PHASES
            if start <= request.token_step <= end
        )
        events.append({
            "request_order": order,
            "key": request.key,
            "layer_index": request.layer,
            "expert_id": request.expert,
            "token_step": request.token_step,
            "phase": phase,
        })
    return events


def _run_synthetic_trial(
    events: list[dict],
    policy: str,
    expert_bytes: int,
    budget_mib: int,
    alpha: float | None,
    seed: int,
    workload: str,
    scenario: str,
) -> tuple[list[dict], list[dict]]:
    capacity = budget_mib * MIB_BYTES // expert_bytes
    cache = _cache(policy, capacity, alpha)
    warmup = 0
    if scenario == "persistent_after_warmup":
        for event in events:
            cache.request(event["key"])
        warmup = len(events)
    return replay_sequence(
        events,
        cache,
        policy,
        expert_bytes,
        budget_mib,
        alpha,
        scenario,
        workload,
        seed,
        "not_applicable",
        warmup_requests=warmup,
    )


def tune_decay_on_synthetic():
    sweep_rows = []
    phase_sweep_rows = []
    for workload_name, generator in WORKLOADS:
        for seed in SEEDS:
            events = _synthetic_events(generator(seed=seed))
            for budget in SYNTHETIC_BUDGETS_MIB:
                for alpha in DECAY_ALPHAS:
                    summary, _ = _run_synthetic_trial(
                        events,
                        "DECAYED_LFU",
                        EXPERT_SIZE_MIB * MIB_BYTES,
                        budget,
                        alpha,
                        seed,
                        workload_name,
                        "cold_start",
                    )
                    all_row = next(row for row in summary if row["phase"] == "all")
                    sweep_rows.append({
                        **all_row,
                        "selection_dataset": "synthetic_development_only",
                    })
                    phase_sweep_rows.extend(
                        {
                            **row,
                            "selection_dataset": "synthetic_development_only",
                        }
                        for row in summary
                        if row["phase"] != "all"
                    )

    grouped = defaultdict(list)
    workload_grouped = defaultdict(list)
    for row in sweep_rows:
        grouped[row["decay_alpha"]].append(row["hit_rate"])
        workload_grouped[
            (row["workload"], row["decay_alpha"])
        ].append(row["hit_rate"])
    selection_rows = [
        {
            "decay_alpha": alpha,
            "trial_count": len(grouped[alpha]),
            "mean_hit_rate": statistics.fmean(grouped[alpha]),
            "selection_metric": (
                "Macro-average hit rate across four synthetic workloads, "
                "five seeds, and six budgets (cold-start only)."
            ),
        }
        for alpha in DECAY_ALPHAS
    ]
    chosen = max(
        selection_rows,
        key=lambda row: (row["mean_hit_rate"], row["decay_alpha"]),
    )
    per_workload = [
        {
            "workload": workload,
            "decay_alpha": alpha,
            "mean_hit_rate": statistics.fmean(
                workload_grouped[(workload, alpha)]
            ),
        }
        for workload, _ in WORKLOADS
        for alpha in DECAY_ALPHAS
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        RESULTS_DIR / "synthetic_decay_sweep.csv",
        list(sweep_rows[0].keys()),
        sweep_rows,
    )
    write_csv(
        RESULTS_DIR / "synthetic_decay_selection.csv",
        list(selection_rows[0].keys()),
        selection_rows,
    )
    write_csv(
        RESULTS_DIR / "synthetic_decay_by_workload.csv",
        list(per_workload[0].keys()),
        per_workload,
    )
    write_csv(
        RESULTS_DIR / "synthetic_decay_phase_sweep.csv",
        list(phase_sweep_rows[0].keys()),
        phase_sweep_rows,
    )
    frozen = {
        "policy": "DECAYED_LFU",
        "decay_alpha": chosen["decay_alpha"],
        "selection_dataset": "Day 3 synthetic workloads only",
        "selection_metric": chosen["selection_metric"],
        "trial_count": chosen["trial_count"],
        "mean_hit_rate": chosen["mean_hit_rate"],
        "candidate_coefficients": list(DECAY_ALPHAS),
        "tie_break": "Prefer the larger alpha when macro-average hit rates tie.",
        "frozen_before_day6_evaluation": True,
    }
    (RESULTS_DIR / "frozen_policy_settings.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        float(chosen["decay_alpha"]),
        sweep_rows,
        phase_sweep_rows,
        selection_rows,
    )


def compare_synthetic(chosen_alpha: float):
    comparison_rows = []
    phase_rows = []
    for workload_name, generator in WORKLOADS:
        for seed in SEEDS:
            events = _synthetic_events(generator(seed=seed))
            for budget in SYNTHETIC_BUDGETS_MIB:
                for scenario in (
                    "cold_start",
                    "persistent_after_warmup",
                ):
                    for policy in POLICY_ORDER:
                        alpha = chosen_alpha if policy == "DECAYED_LFU" else None
                        summary, _ = _run_synthetic_trial(
                            events,
                            policy,
                            EXPERT_SIZE_MIB * MIB_BYTES,
                            budget,
                            alpha,
                            seed,
                            workload_name,
                            scenario,
                        )
                        comparison_rows.extend(summary)
                        phase_rows.extend(
                            row for row in summary if row["phase"] != "all"
                        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        RESULTS_DIR / "synthetic_policy_comparison.csv",
        list(comparison_rows[0].keys()),
        comparison_rows,
    )
    write_csv(
        RESULTS_DIR / "synthetic_phase_comparison.csv",
        list(phase_rows[0].keys()),
        phase_rows,
    )
    return comparison_rows


def _day6_expert_bytes(config: dict) -> int:
    summary, _ = expert_storage_from_safetensors(
        DAY6_EXPERTS_PATH,
        config,
    )
    return int(summary["expert_weight_bytes_per_layer_expert"])


def compare_day6(chosen_alpha: float):
    with DAY6_REPORT_PATH.open(encoding="utf-8") as handle:
        pilot_report = json.load(handle)
    if pilot_report.get("status") != "complete":
        raise RuntimeError(
            f"Day 6 pilot is not complete: {pilot_report.get('status')}"
        )
    rows = read_day6_trace()
    config = load_config()
    trace_validation = validate_day6_trace(rows, config)
    expert_bytes = _day6_expert_bytes(config)
    prompt_rows = {
        category: sorted(
            (
                {
                    **row,
                    "key": f"L{row['layer_index']}:E{row['expert_id']}",
                }
                for row in rows
                if row["prompt_category"] == category
            ),
            key=lambda row: row["request_order"],
        )
        for category in CATEGORY_ORDER
    }

    comparison_rows = []
    per_layer_rows = []
    for scenario in ("reset_each_prompt", "persistent_across_prompts"):
        for budget in CACHE_BUDGETS_MIB:
            capacity = budget * MIB_BYTES // expert_bytes
            for policy in POLICY_ORDER:
                persistent_cache = _cache(
                    policy,
                    capacity,
                    chosen_alpha if policy == "DECAYED_LFU" else None,
                )
                prompt_summaries = []
                prompt_layers = []
                for category in CATEGORY_ORDER:
                    cache = (
                        _cache(
                            policy,
                            capacity,
                            chosen_alpha if policy == "DECAYED_LFU" else None,
                        )
                        if scenario == "reset_each_prompt"
                        else persistent_cache
                    )
                    summary, layers = replay_sequence(
                        prompt_rows[category],
                        cache,
                        policy,
                        expert_bytes,
                        budget,
                        chosen_alpha if policy == "DECAYED_LFU" else None,
                        scenario,
                        "day6_real_trace",
                        "not_applicable",
                        category,
                    )
                    prompt_summaries.extend(summary)
                    prompt_layers.extend(layers)
                comparison_rows.extend(prompt_summaries)
                per_layer_rows.extend(prompt_layers)
                for phase in ("prefill", "decode", "all"):
                    selected = [
                        row for row in prompt_summaries
                        if row["phase"] == phase
                    ]
                    requests = sum(row["requests"] for row in selected)
                    hits = sum(row["hits"] for row in selected)
                    misses = sum(row["misses"] for row in selected)
                    evictions = sum(row["evictions"] for row in selected)
                    # The cache occupancy aggregate is intentionally blank:
                    # reset caches are separate, while phase totals span
                    # non-contiguous requests in the persistent scenario.
                    state = (
                        _estimated_state(
                            persistent_cache,
                            policy,
                            _entry_count(persistent_cache),
                        )
                        if scenario == "persistent_across_prompts"
                        else {
                            "resident_policy_metadata_fields_per_entry":
                                METADATA_FIELDS[policy],
                            "estimated_resident_metadata_bytes": "",
                            "estimated_global_metadata_bytes": "",
                            "estimated_policy_metadata_bytes": "",
                            "state_overhead_model": (
                                "Per-prompt cache states are separate; "
                                "aggregate state is not applicable."
                            ),
                        }
                    )
                    comparison_rows.append({
                        "scenario": scenario,
                        "workload": "day6_real_trace",
                        "seed": "not_applicable",
                        "prompt_category": "ALL_PROMPTS",
                        "cache_budget_mib": budget,
                        "policy": policy,
                        "decay_alpha": (
                            chosen_alpha if policy == "DECAYED_LFU" else ""
                        ),
                        "phase": phase,
                        "requests": requests,
                        "hits": hits,
                        "misses": misses,
                        "hit_rate": hits / requests if requests else 0.0,
                        "miss_rate": misses / requests if requests else 0.0,
                        "modeled_expert_bytes_transferred": misses * expert_bytes,
                        "evictions": evictions,
                        "cache_capacity_experts": (
                            capacity if policy != "NO_CACHE" else 0
                        ),
                        "cache_occupancy_bytes_at_boundary": "",
                        "peak_resident_experts": "",
                        **state,
                        "warmup_requests_excluded": 0,
                        "traffic_is_modeled_not_measured": True,
                    })

                for phase in ("prefill", "decode", "all"):
                    selected = [
                        row for row in prompt_layers
                        if row["phase"] == phase
                    ]
                    for layer in range(int(config["num_hidden_layers"])):
                        layer_selected = [
                            row for row in selected
                            if row["layer_index"] == layer
                        ]
                        requests = sum(row["requests"] for row in layer_selected)
                        hits = sum(row["hits"] for row in layer_selected)
                        misses = sum(row["misses"] for row in layer_selected)
                        per_layer_rows.append({
                            "scenario": scenario,
                            "workload": "day6_real_trace",
                            "seed": "not_applicable",
                            "prompt_category": "ALL_PROMPTS",
                            "cache_budget_mib": budget,
                            "policy": policy,
                            "decay_alpha": (
                                chosen_alpha if policy == "DECAYED_LFU" else ""
                            ),
                            "phase": phase,
                            "layer_index": layer,
                            "requests": requests,
                            "hits": hits,
                            "misses": misses,
                            "hit_rate": hits / requests if requests else 0.0,
                            "miss_rate": misses / requests if requests else 0.0,
                            "modeled_expert_bytes_transferred": (
                                misses * expert_bytes
                            ),
                            "evictions": sum(
                                row["evictions"] for row in layer_selected
                            ),
                            "cache_occupancy_bytes_at_boundary": "",
                            "estimated_policy_metadata_bytes": "",
                            "traffic_is_modeled_not_measured": True,
                        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        RESULTS_DIR / "day6_policy_comparison.csv",
        list(comparison_rows[0].keys()),
        comparison_rows,
    )
    write_csv(
        RESULTS_DIR / "day6_policy_per_layer.csv",
        list(per_layer_rows[0].keys()),
        per_layer_rows,
    )
    return (
        comparison_rows,
        per_layer_rows,
        trace_validation,
        expert_bytes,
        pilot_report,
    )


def _relative_to_lru(rows: list[dict], grouping: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(dict)
    for row in rows:
        if row["phase"] != "all":
            continue
        key = tuple(row[field] for field in grouping)
        groups[key][row["policy"]] = row
    result = []
    for key, policies in groups.items():
        lru = policies.get("LRU")
        if lru is None:
            continue
        for policy in ("LFU", "DECAYED_LFU", "SLRU", "NO_CACHE"):
            candidate = policies.get(policy)
            if candidate is None:
                continue
            result.append({
                **dict(zip(grouping, key)),
                "policy": policy,
                "lru_miss_rate": lru["miss_rate"],
                "policy_miss_rate": candidate["miss_rate"],
                "miss_rate_delta_vs_lru": (
                    candidate["miss_rate"] - lru["miss_rate"]
                ),
                "policy_hits_minus_lru": candidate["hits"] - lru["hits"],
                "policy_evictions": candidate.get(
                    "evictions",
                    candidate.get("mean_evictions", ""),
                ),
            })
    return result


def _aggregate_synthetic(comparison_rows):
    grouped = defaultdict(list)
    for row in comparison_rows:
        if row["phase"] == "all":
            grouped[
                (row["scenario"], row["workload"], row["cache_budget_mib"],
                 row["policy"])
            ].append(row)
    aggregate_rows = []
    for key, values in sorted(grouped.items()):
        requests = sum(row["requests"] for row in values)
        hits = sum(row["hits"] for row in values)
        misses = sum(row["misses"] for row in values)
        aggregate_rows.append({
            "scenario": key[0],
            "workload": key[1],
            "cache_budget_mib": key[2],
            "policy": key[3],
            "phase": "all",
            "seed_count": len(values),
            "requests": requests,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / requests if requests else 0.0,
            "miss_rate": misses / requests if requests else 0.0,
            "modeled_expert_bytes_transferred": sum(
                row["modeled_expert_bytes_transferred"] for row in values
            ),
            "mean_evictions": statistics.fmean(
                row["evictions"] for row in values
            ),
            "mean_estimated_policy_metadata_bytes": statistics.fmean(
                row["estimated_policy_metadata_bytes"] for row in values
            ),
        })
    return aggregate_rows


def make_figures(sweep_rows, synthetic_rows, real_rows) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for workload, _ in WORKLOADS:
        selected = []
        for alpha in DECAY_ALPHAS:
            values = [
                row["hit_rate"] for row in sweep_rows
                if row["workload"] == workload
                and row["decay_alpha"] == alpha
            ]
            selected.append(statistics.fmean(values) if values else 0.0)
        ax.plot(DECAY_ALPHAS, selected, marker="o", label=workload)
    ax.set_xlabel("Per-request decay coefficient (alpha)")
    ax.set_ylabel("Mean synthetic cold-start hit rate")
    ax.set_title("Synthetic-only decayed-LFU coefficient sweep")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "decay_coefficient_sweep.png", dpi=160)
    plt.close(fig)

    aggregate = _aggregate_synthetic(synthetic_rows)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    for row_index, (workload, _) in enumerate(WORKLOADS):
        for column_index, scenario in enumerate(
            ("cold_start", "persistent_after_warmup")
        ):
            ax = axes[row_index][column_index]
            for policy in POLICY_ORDER:
                selected = sorted(
                    (
                        row for row in aggregate
                        if row["scenario"] == scenario
                        and row["workload"] == workload
                        and row["policy"] == policy
                    ),
                    key=lambda row: row["cache_budget_mib"],
                )
                ax.plot(
                    [row["cache_budget_mib"] for row in selected],
                    [row["miss_rate"] for row in selected],
                    marker="o",
                    label=policy,
                )
            ax.set_title(f"{workload}: {scenario}")
            ax.set_xlabel("Budget (MiB)")
            ax.set_ylabel("Mean miss rate")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "synthetic_policy_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (scenario, phase) in zip(
        axes.flat,
        (
            ("reset_each_prompt", "prefill"),
            ("reset_each_prompt", "decode"),
            ("persistent_across_prompts", "prefill"),
            ("persistent_across_prompts", "decode"),
        ),
    ):
        for policy in POLICY_ORDER:
            selected = sorted(
                (
                    row for row in real_rows
                    if row["scenario"] == scenario
                    and row["prompt_category"] == "ALL_PROMPTS"
                    and row["phase"] == phase
                    and row["policy"] == policy
                ),
                key=lambda row: row["cache_budget_mib"],
            )
            ax.plot(
                [row["cache_budget_mib"] for row in selected],
                [row["miss_rate"] for row in selected],
                marker="o",
                label=policy,
            )
        ax.set_title(f"{scenario.replace('_', ' ')} — {phase}")
        ax.set_xlabel("Budget (MiB)")
        ax.set_ylabel("Miss rate")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Exploratory Day 6 real-trace replay")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "day6_policy_comparison.png", dpi=160)
    plt.close(fig)


def _write_findings(
    chosen_alpha,
    selection_rows,
    synthetic_comparison,
    real_comparison,
    expert_bytes,
    validation,
    pilot_report,
):
    synthetic_aggregate = _aggregate_synthetic(synthetic_comparison)
    synthetic_rel = _relative_to_lru(
        synthetic_aggregate,
        ("scenario", "workload", "cache_budget_mib"),
    )
    real_all = [
        row for row in real_comparison
        if row["prompt_category"] == "ALL_PROMPTS"
        and row["phase"] == "all"
    ]
    real_rel = _relative_to_lru(
        real_all,
        ("scenario", "cache_budget_mib"),
    )
    write_csv(
        RESULTS_DIR / "synthetic_relative_to_lru.csv",
        list(synthetic_rel[0].keys()),
        synthetic_rel,
    )
    write_csv(
        RESULTS_DIR / "day6_relative_to_lru.csv",
        list(real_rel[0].keys()),
        real_rel,
    )

    alpha_summary = next(
        row for row in selection_rows
        if row["decay_alpha"] == chosen_alpha
    )
    policy_summary = defaultdict(list)
    for row in synthetic_rel:
        policy_summary[(row["workload"], row["policy"])].append(
            row["miss_rate_delta_vs_lru"]
        )
    workload_notes = []
    for workload, _ in WORKLOADS:
        parts = []
        for policy in ("LFU", "DECAYED_LFU", "SLRU"):
            values = policy_summary[(workload, policy)]
            mean_delta = statistics.fmean(values) if values else float("nan")
            wins = sum(value < 0 for value in values)
            parts.append(
                f"{policy} mean miss-rate delta vs LRU {mean_delta:+.4f} "
                f"({wins}/{len(values)} workload/seed-budget-scenario cells better)"
            )
        workload_notes.append(f"- **{workload}:** " + "; ".join(parts) + ".")

    real_notes = []
    phase_notes = []
    real_detail_notes = []
    for scenario in ("reset_each_prompt", "persistent_across_prompts"):
        for budget in CACHE_BUDGETS_MIB:
            selected = {
                row["policy"]: row for row in real_all
                if row["scenario"] == scenario
                and row["cache_budget_mib"] == budget
            }
            if "LRU" not in selected:
                continue
            real_notes.append(
                f"- {scenario}, {budget} MiB: LRU "
                f"{selected['LRU']['miss_rate']:.4f}; decayed LFU "
                f"{selected['DECAYED_LFU']['miss_rate']:.4f} "
                f"(delta {selected['DECAYED_LFU']['miss_rate'] - selected['LRU']['miss_rate']:+.4f}); "
                f"SLRU {selected['SLRU']['miss_rate']:.4f} "
                f"(delta {selected['SLRU']['miss_rate'] - selected['LRU']['miss_rate']:+.4f})."
            )
            if budget == 256:
                for phase in ("prefill", "decode"):
                    phase_selected = {
                        row["policy"]: row for row in real_comparison
                        if row["scenario"] == scenario
                        and row["prompt_category"] == "ALL_PROMPTS"
                        and row["cache_budget_mib"] == budget
                        and row["phase"] == phase
                    }
                    phase_notes.append(
                        f"- {scenario}, 256 MiB, {phase}: LRU "
                        f"{phase_selected['LRU']['miss_rate']:.4f}, decayed LFU "
                        f"{phase_selected['DECAYED_LFU']['miss_rate']:.4f}, "
                        f"SLRU {phase_selected['SLRU']['miss_rate']:.4f}."
                    )
            if budget in (256, 512):
                detail_selected = {
                    row["policy"]: row for row in real_all
                    if row["scenario"] == scenario
                    and row["cache_budget_mib"] == budget
                }
                real_detail_notes.append(
                    f"- {scenario}, {budget} MiB — "
                    + "; ".join(
                        f"{policy}: {detail_selected[policy]['misses']} misses, "
                        f"{detail_selected[policy]['modeled_expert_bytes_transferred']:,} "
                        f"modeled bytes, {detail_selected[policy]['evictions']} "
                        "evictions"
                        for policy in ("LRU", "DECAYED_LFU", "SLRU")
                    )
                    + "."
                )

    persistent_512 = {
        row["policy"]: row for row in real_all
        if row["scenario"] == "persistent_across_prompts"
        and row["cache_budget_mib"] == 512
    }

    prompt_cache_notes = []
    for scenario in ("reset_each_prompt", "persistent_across_prompts"):
        for category in CATEGORY_ORDER:
            selected = {
                row["policy"]: row for row in real_comparison
                if row["scenario"] == scenario
                and row["prompt_category"] == category
                and row["cache_budget_mib"] == 256
                and row["phase"] == "all"
            }
            prompt_cache_notes.append(
                f"- {scenario}, {category}: LRU "
                f"{selected['LRU']['miss_rate']:.4f}, decayed LFU "
                f"{selected['DECAYED_LFU']['miss_rate']:.4f}, SLRU "
                f"{selected['SLRU']['miss_rate']:.4f}."
            )

    prompt_lines = []
    request_summaries = validation["request_summaries"]
    for report in pilot_report["completed_requests"]:
        summary = next(
            row for row in request_summaries
            if row["request_id"] == report["request_id"]
        )
        prompt_lines.append(
            f"- {report['prompt_category']}: "
            f"{summary['prefill_events']} prefill + "
            f"{summary['decode_events']} decode selections; "
            f"{report['prompt_tokens']} input tokens, "
            f"{report['generated_tokens']} generated tokens."
        )

    text = [
        "# Day 7 cache-policy investigation",
        "",
        "## Method",
        "",
        f"Day 6 trace validation passed for {validation['event_count']} genuine "
        f"top-4 selections across four requests and 20 MoE layers. Expert size "
        f"is {expert_bytes:,} bytes per layer-expert, derived from the pinned "
        "Safetensors tensor shapes and BF16 dtype. Policies use identical "
        "layer-qualified keys, event order, budgets, and expert sizes.",
        "",
        "Decay coefficients were selected only on Day 3 synthetic development "
        "workloads (static skew, abrupt changes, gradual drift, alternating "
        "tasks), seeds 42–46, six budgets, and cold-start replay. The chosen "
        f"alpha was **{chosen_alpha:g}**, with macro-average hit rate "
        f"{alpha_summary['mean_hit_rate']:.4f} over "
        f"{alpha_summary['trial_count']} workload/seed/budget trials. This "
        "setting was frozen before Day 6 evaluation.",
        "",
        "Decayed LFU lazily multiplies resident frequency by "
        "`alpha ** elapsed_requests` before a hit update or eviction comparison; "
        "the current hit adds one, and eviction chooses the smallest aged score "
        "then least-recently-used entry. SLRU uses a 25% probationary segment; "
        "new entries enter probationary and a repeat promotes them to protected.",
        "",
        "Day 3 persistent-cache results use one complete trace pass as an "
        "unmeasured warm-up, then measure a second identical pass without "
        "clearing state. Day 6 reset results start empty at each prompt; the "
        "persistent Day 6 cache starts empty and carries state across the four "
        "requests in their recorded order. All retain state from prefill to "
        "decode within a request.",
        "",
        "## Day 6 prompt coverage",
        "",
        *prompt_lines,
        "",
        "## Synthetic ablation",
        "",
        *workload_notes,
        "",
        "## Exploratory Day 6 replay",
        "",
        *real_notes,
        "",
        "At 256 MiB, phase-separated miss rates were:",
        "",
        *phase_notes,
        "",
        "Representative total misses, modeled bytes, and evictions:",
        "",
        *real_detail_notes,
        "",
        "At the end of the persistent 512 MiB replay, estimated fixed-width "
        "policy metadata was "
        + ", ".join(
            f"{policy} {persistent_512[policy]['estimated_policy_metadata_bytes']:,} B"
            for policy in ("LRU", "LFU", "DECAYED_LFU", "SLRU")
        )
        + ". Key strings, containers, and Python runtime overhead are excluded.",
        "",
        "At 256 MiB, per-prompt miss rates were:",
        "",
        *prompt_cache_notes,
        "",
        "## Limitations",
        "",
        "- Day 6 is four short prompts; the trace is tiny and already informed our research design. Real-trace policy results are exploratory, not held-out validation.",
        "- Prompt lengths differ. The general-knowledge request ended after four generated tokens, so its decode sample is shorter.",
        "- Prefill routes tokens as a batch, while replay serializes layer/token/rank events. Actual hardware may coalesce or issue expert loads concurrently.",
        "- Modeled expert bytes are misses multiplied by tensor-derived expert size, not measured memory or disk traffic. No inference speedup or tokens-per-second claim is made.",
        "- State-overhead estimates count fixed-width policy metadata fields only; keys, containers, Python object overhead, and allocator costs are excluded.",
        "- The tokenizer emitted a regex-pattern warning about possible incorrect tokenization. This run was not repeated with a changed tokenizer setting.",
        "- Synthetic workloads are constructed distributions, not real model traces; results depend on their sampling and shift assumptions.",
        "",
        "A later held-out validation should use a newly approved, preregistered prompt set spanning more tasks and lengths, with fixed order and decoding settings, separate reset/persistent cache runs, and no coefficient retuning on those traces. It should keep the same prefill-serialization caveat and report measured system effects separately from cache replay.",
    ]
    (RESULTS_DIR / "research_findings.md").write_text(
        "\n".join(text) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (
        chosen_alpha,
        sweep_rows,
        phase_sweep_rows,
        selection_rows,
    ) = tune_decay_on_synthetic()
    # Freeze is written by the development-only selector before real replay.
    with (RESULTS_DIR / "frozen_policy_settings.json").open(
        encoding="utf-8"
    ) as handle:
        frozen = json.load(handle)
    if frozen["decay_alpha"] != chosen_alpha:
        raise RuntimeError("Frozen decay coefficient changed unexpectedly")

    synthetic_comparison = compare_synthetic(chosen_alpha)
    (
        real_comparison,
        real_layer_rows,
        real_validation,
        expert_bytes,
        pilot_report,
    ) = compare_day6(chosen_alpha)
    make_figures(sweep_rows, synthetic_comparison, real_comparison)
    _write_findings(
        chosen_alpha,
        selection_rows,
        synthetic_comparison,
        real_comparison,
        expert_bytes,
        real_validation,
        pilot_report,
    )
    summary = {
        "status": "complete",
        "frozen_policy_settings": frozen,
        "day6_trace_validation": real_validation,
        "day6_event_count": real_validation["event_count"],
        "day6_layer_expert_size_bytes": expert_bytes,
        "synthetic_development": {
            "workloads": [name for name, _ in WORKLOADS],
            "seeds": list(SEEDS),
            "budgets_mib": list(SYNTHETIC_BUDGETS_MIB),
            "decay_coefficients": list(DECAY_ALPHAS),
            "selection_uses_real_trace": False,
        },
        "evaluation": {
            "policies": list(POLICY_ORDER),
            "day6_scenarios": [
                "reset_each_prompt",
                "persistent_across_prompts",
            ],
            "synthetic_scenarios": [
                "cold_start",
                "persistent_after_warmup",
            ],
            "real_trace_is_exploratory": True,
            "additional_inference_run": False,
            "transfer_bytes_measured": False,
        },
        "limitations": [
            "Day 6 is a small, design-informed four-prompt sample.",
            "Prefill is batched but replay is serialized by recorded layer/token/rank order.",
            "Synthetic workloads are constructed routing distributions.",
            "Policy state size is a fixed-width metadata estimate, not Python heap measurement.",
            "The tokenizer emitted a possible incorrect-tokenization regex warning; no rerun was performed.",
        ],
        "output_files": [
            "synthetic_decay_sweep.csv",
            "synthetic_decay_selection.csv",
            "synthetic_decay_by_workload.csv",
            "synthetic_decay_phase_sweep.csv",
            "frozen_policy_settings.json",
            "synthetic_policy_comparison.csv",
            "synthetic_phase_comparison.csv",
            "synthetic_relative_to_lru.csv",
            "day6_policy_comparison.csv",
            "day6_policy_per_layer.csv",
            "day6_relative_to_lru.csv",
            "research_findings.md",
        ],
    }
    (RESULTS_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    real_all_rows = [
        row for row in real_comparison
        if row["prompt_category"] == "ALL_PROMPTS"
        and row["phase"] == "all"
    ]
    print(
        f"Frozen synthetic-development alpha: {chosen_alpha:g}; "
        f"validated Day 6 events: {real_validation['event_count']}."
    )
    for scenario in ("reset_each_prompt", "persistent_across_prompts"):
        print(scenario)
        for budget in CACHE_BUDGETS_MIB:
            selected = {
                row["policy"]: row for row in real_all_rows
                if row["scenario"] == scenario
                and row["cache_budget_mib"] == budget
            }
            print(
                f"  {budget:>3} MiB "
                + " ".join(
                    f"{policy}={selected[policy]['miss_rate']:.4f}"
                    for policy in POLICY_ORDER
                )
            )
    print("Generated Day 7 outputs under results/day7/ and figures/day7/.")


if __name__ == "__main__":
    main()
