"""Analyze Day 6 MobileMoE prompt traces and replay cache policies."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from baseline import LFUCache, LRUCache, NoCache
from day5_analysis import (
    CACHE_BUDGETS_MIB,
    MIB_BYTES,
    analyze_usage,
    expert_storage_from_safetensors,
    load_config,
    validate_trace,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
TRACE_PATH = ROOT / "results" / "day6" / "routing_events.csv"
REPORT_PATH = ROOT / "results" / "day6" / "pilot_report.json"
RESULTS_DIR = ROOT / "results" / "day6"
FIGURES_DIR = ROOT / "figures" / "day6"
CATEGORY_ORDER = (
    "coding",
    "mathematics",
    "general_knowledge",
    "scientific_reasoning",
)


def read_trace(path: Path = TRACE_PATH) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    integer_fields = (
        "request_order",
        "forward_index",
        "phase_token_index",
        "token_index",
        "layer_index",
        "expert_id",
        "selected_rank",
        "seed",
        "max_new_tokens",
    )
    for row in rows:
        for field in integer_fields:
            row[field] = int(row[field])
        row["do_sample"] = row["do_sample"].lower() == "true"
    return rows


def validate_day6_trace(rows: list[dict], config: dict) -> dict:
    if not rows:
        raise ValueError("Day 6 routing trace is empty")
    if [row["request_order"] for row in rows] != list(range(len(rows))):
        raise ValueError("Global request_order is not contiguous")
    categories = list(dict.fromkeys(row["prompt_category"] for row in rows))
    if tuple(categories) != CATEGORY_ORDER:
        raise ValueError(f"Unexpected prompt order: {categories}")

    per_request = defaultdict(list)
    for row in rows:
        per_request[row["request_id"]].append(row)
    if len(per_request) != len(CATEGORY_ORDER):
        raise ValueError("Expected exactly four request IDs")

    request_summaries = []
    seen_categories = []
    for request_id, request_rows in per_request.items():
        category = request_rows[0]["prompt_category"]
        if any(row["prompt_category"] != category for row in request_rows):
            raise ValueError(f"Request {request_id} has mixed categories")
        if category in seen_categories:
            raise ValueError(f"Category {category} appears in multiple requests")
        seen_categories.append(category)
        local_rows = [dict(row) for row in request_rows]
        for local_order, row in enumerate(local_rows):
            row["request_order"] = local_order
        summary = validate_trace(local_rows, config)
        request_summaries.append({
            "request_id": request_id,
            "prompt_category": category,
            **summary,
        })

    if tuple(seen_categories) != CATEGORY_ORDER:
        raise ValueError("Request order does not match the fixed pilot order")
    return {
        "event_count": len(rows),
        "request_count": len(request_summaries),
        "prompt_categories": categories,
        "request_summaries": request_summaries,
        "global_request_order_valid": True,
        "per_request_top_k_and_layer_mapping_valid": True,
    }


def analyze_prompt_routing(rows: list[dict], config: dict):
    frequency_rows = []
    metric_rows = []
    reuse_rows = []
    reuse_distribution_rows = []
    grouped = {
        category: [
            row for row in rows if row["prompt_category"] == category
        ]
        for category in CATEGORY_ORDER
    }
    for category, category_rows in grouped.items():
        metrics, frequencies, reuse, distribution, _ = analyze_usage(
            category_rows, config
        )
        metric_rows.extend(
            {"prompt_category": category, **row} for row in metrics
        )
        frequency_rows.extend(
            {"prompt_category": category, **row} for row in frequencies
        )
        reuse_rows.extend(
            {"prompt_category": category, **row} for row in reuse
        )
        reuse_distribution_rows.extend(
            {"prompt_category": category, **row} for row in distribution
        )
    return (
        metric_rows,
        frequency_rows,
        reuse_rows,
        reuse_distribution_rows,
    )


def prompt_transition_overlap(rows: list[dict]) -> list[dict]:
    result = []
    layer_ids = sorted({row["layer_index"] for row in rows})
    for previous_category, current_category in zip(
        CATEGORY_ORDER, CATEGORY_ORDER[1:]
    ):
        for phase in ("prefill", "decode"):
            for layer in layer_ids:
                previous = Counter(
                    row["expert_id"] for row in rows
                    if row["prompt_category"] == previous_category
                    and row["phase"] == phase
                    and row["layer_index"] == layer
                )
                current = Counter(
                    row["expert_id"] for row in rows
                    if row["prompt_category"] == current_category
                    and row["phase"] == phase
                    and row["layer_index"] == layer
                )
                previous_set = set(previous)
                current_set = set(current)
                union = previous_set | current_set
                dot = sum(
                    previous[expert] * current[expert]
                    for expert in previous_set | current_set
                )
                previous_norm = math.sqrt(sum(
                    count * count for count in previous.values()
                ))
                current_norm = math.sqrt(sum(
                    count * count for count in current.values()
                ))
                result.append({
                    "previous_prompt_category": previous_category,
                    "current_prompt_category": current_category,
                    "phase": phase,
                    "layer_index": layer,
                    "previous_selection_count": sum(previous.values()),
                    "current_selection_count": sum(current.values()),
                    "previous_unique_experts": len(previous_set),
                    "current_unique_experts": len(current_set),
                    "shared_experts": len(previous_set & current_set),
                    "unique_set_jaccard": (
                        len(previous_set & current_set) / len(union)
                        if union else 0.0
                    ),
                    "frequency_cosine_similarity": (
                        dot / (previous_norm * current_norm)
                        if previous_norm and current_norm else 0.0
                    ),
                })
    return result


def _cache(policy: str, capacity: int):
    if policy == "NO_CACHE":
        return NoCache(budget_mib=0, expert_size_mib=1)
    if policy == "LRU":
        return LRUCache(budget_mib=capacity, expert_size_mib=1)
    if policy == "LFU":
        return LFUCache(budget_mib=capacity, expert_size_mib=1)
    raise ValueError(f"Unknown cache policy: {policy}")


def _entry_count(cache) -> int:
    return len(getattr(cache, "entries", {}))


def _replay_prompt(
    cache,
    rows: list[dict],
    capacity: int,
    expert_bytes: int,
    scenario: str,
    budget_mib: int,
    policy: str,
    category: str,
):
    phases = ("prefill", "decode")
    phase_counts = {
        phase: {"hits": 0, "misses": 0, "evictions": 0}
        for phase in phases
    }
    layer_counts = defaultdict(
        lambda: {"hits": 0, "misses": 0, "evictions": 0}
    )
    phase_occupancy = {}

    for position, event in enumerate(rows):
        phase = event["phase"]
        layer = event["layer_index"]
        before = _entry_count(cache)
        hit = cache.request(f"L{layer}:E{event['expert_id']}")
        layer_count = layer_counts[(phase, layer)]
        if hit:
            phase_counts[phase]["hits"] += 1
            layer_count["hits"] += 1
        else:
            phase_counts[phase]["misses"] += 1
            layer_count["misses"] += 1
            if policy != "NO_CACHE" and before >= capacity:
                phase_counts[phase]["evictions"] += 1
                layer_count["evictions"] += 1

        next_phase = (
            rows[position + 1]["phase"]
            if position + 1 < len(rows) else None
        )
        if next_phase != phase:
            phase_occupancy[phase] = _entry_count(cache)

    def summary_row(phase, counts, occupancy):
        requests = counts["hits"] + counts["misses"]
        return {
            "scenario": scenario,
            "cache_budget_mib": budget_mib,
            "policy": policy,
            "prompt_category": category,
            "phase": phase,
            "requests": requests,
            "hits": counts["hits"],
            "misses": counts["misses"],
            "miss_rate": counts["misses"] / requests if requests else 0.0,
            "evictions": counts["evictions"],
            "expert_weight_bytes_per_entry": expert_bytes,
            "cache_capacity_experts": capacity if policy != "NO_CACHE" else 0,
            "cache_occupancy_bytes_at_boundary": (
                occupancy * expert_bytes if policy != "NO_CACHE" else 0
            ),
            "modeled_expert_weight_bytes_on_miss": (
                counts["misses"] * expert_bytes
            ),
            "traffic_is_modeled_not_measured": True,
        }

    summary = [
        summary_row(
            phase,
            phase_counts[phase],
            phase_occupancy.get(phase, _entry_count(cache)),
        )
        for phase in phases
    ]
    all_counts = {
        name: sum(phase_counts[phase][name] for phase in phases)
        for name in ("hits", "misses", "evictions")
    }
    summary.append(summary_row("all", all_counts, _entry_count(cache)))

    layer_rows = []
    for phase in ("prefill", "decode", "all"):
        for layer in sorted({row["layer_index"] for row in rows}):
            if phase == "all":
                counts = {
                    name: (
                        layer_counts[("prefill", layer)][name]
                        + layer_counts[("decode", layer)][name]
                    )
                    for name in ("hits", "misses", "evictions")
                }
            else:
                counts = layer_counts[(phase, layer)]
            requests = counts["hits"] + counts["misses"]
            layer_rows.append({
                "scenario": scenario,
                "cache_budget_mib": budget_mib,
                "policy": policy,
                "prompt_category": category,
                "phase": phase,
                "layer_index": layer,
                "requests": requests,
                "hits": counts["hits"],
                "misses": counts["misses"],
                "miss_rate": counts["misses"] / requests if requests else 0.0,
                "evictions_on_layer_requests": counts["evictions"],
                "modeled_expert_weight_bytes_on_miss": (
                    counts["misses"] * expert_bytes
                ),
            })
    return summary, layer_rows


def replay_scenarios(
    rows: list[dict],
    expert_bytes: int,
    budgets: tuple[int, ...] = CACHE_BUDGETS_MIB,
):
    summary_rows = []
    layer_rows = []
    categories = list(CATEGORY_ORDER)
    policies = ("NO_CACHE", "LRU", "LFU")

    for scenario in ("reset_each_prompt", "persistent_across_prompts"):
        for budget_mib in budgets:
            budget_bytes = budget_mib * MIB_BYTES
            capacity = budget_bytes // expert_bytes
            for policy in policies:
                persistent_cache = _cache(policy, capacity)
                prompt_summaries = []
                prompt_layers = []
                for category in categories:
                    request_rows = sorted(
                        (
                            row for row in rows
                            if row["prompt_category"] == category
                        ),
                        key=lambda row: row["request_order"],
                    )
                    active_cache = (
                        _cache(policy, capacity)
                        if scenario == "reset_each_prompt"
                        else persistent_cache
                    )
                    one_summary, one_layers = _replay_prompt(
                        active_cache,
                        request_rows,
                        capacity,
                        expert_bytes,
                        scenario,
                        budget_mib,
                        policy,
                        category,
                    )
                    prompt_summaries.extend(one_summary)
                    prompt_layers.extend(one_layers)

                summary_rows.extend(prompt_summaries)
                layer_rows.extend(prompt_layers)
                all_request_rows = [
                    row for row in prompt_summaries if row["phase"] == "all"
                ]
                totals = {
                    name: sum(row[name] for row in all_request_rows)
                    for name in ("requests", "hits", "misses", "evictions")
                }
                totals["miss_rate"] = (
                    totals["misses"] / totals["requests"]
                    if totals["requests"] else 0.0
                )
                totals.update({
                    "scenario": scenario,
                    "cache_budget_mib": budget_mib,
                    "policy": policy,
                    "prompt_category": "ALL_PROMPTS",
                    "phase": "all",
                    "expert_weight_bytes_per_entry": expert_bytes,
                    "cache_capacity_experts": (
                        capacity if policy != "NO_CACHE" else 0
                    ),
                    "cache_occupancy_bytes_at_boundary": (
                        _entry_count(persistent_cache) * expert_bytes
                        if scenario == "persistent_across_prompts"
                        and policy != "NO_CACHE"
                        else (0 if policy == "NO_CACHE" else "")
                    ),
                    "modeled_expert_weight_bytes_on_miss": (
                        totals["misses"] * expert_bytes
                    ),
                    "traffic_is_modeled_not_measured": True,
                })
                summary_rows.append(totals)

                for phase in ("prefill", "decode", "all"):
                    for layer in sorted({
                        row["layer_index"] for row in rows
                    }):
                        selected = [
                            row for row in prompt_layers
                            if row["phase"] == phase
                            and row["layer_index"] == layer
                        ]
                        counts = {
                            name: sum(row[name] for row in selected)
                            for name in ("requests", "hits", "misses")
                        }
                        evictions = sum(
                            row["evictions_on_layer_requests"]
                            for row in selected
                        )
                        layer_rows.append({
                            "scenario": scenario,
                            "cache_budget_mib": budget_mib,
                            "policy": policy,
                            "prompt_category": "ALL_PROMPTS",
                            "phase": phase,
                            "layer_index": layer,
                            **counts,
                            "miss_rate": (
                                counts["misses"] / counts["requests"]
                                if counts["requests"] else 0.0
                            ),
                            "evictions_on_layer_requests": evictions,
                            "modeled_expert_weight_bytes_on_miss": (
                                counts["misses"] * expert_bytes
                            ),
                        })
    return summary_rows, layer_rows


def make_figures(frequency_rows, cache_rows, transition_rows) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for phase in ("prefill", "decode"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for category, ax in zip(CATEGORY_ORDER, axes.flat):
            selected = [
                row for row in frequency_rows
                if row["prompt_category"] == category
                and row["phase"] == phase
            ]
            layers = sorted({row["layer_index"] for row in selected})
            experts = sorted({row["expert_id"] for row in selected})
            matrix = [
                [
                    next(
                        row["selection_count"] for row in selected
                        if row["layer_index"] == layer
                        and row["expert_id"] == expert
                    )
                    for expert in experts
                ]
                for layer in layers
            ]
            image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
            ax.set_title(category.replace("_", " "))
            ax.set_xlabel("Expert ID")
            ax.set_ylabel("MoE layer")
            ax.set_xticks(range(0, len(experts), 5))
            ax.set_xticklabels(experts[::5])
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers)
        fig.suptitle(f"MobileMoE expert selections — {phase}")
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Selections")
        fig.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.91,
                            hspace=0.28, wspace=0.25)
        fig.savefig(
            FIGURES_DIR / f"expert_routing_heatmap_{phase}.png",
            dpi=160,
        )
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, scenario in zip(
        axes, ("reset_each_prompt", "persistent_across_prompts")
    ):
        for policy in ("NO_CACHE", "LRU", "LFU"):
            selected = sorted(
                (
                    row for row in cache_rows
                    if row["scenario"] == scenario
                    and row["prompt_category"] == "ALL_PROMPTS"
                    and row["phase"] == "all"
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
        ax.set_title(scenario.replace("_", " "))
        ax.set_xlabel("Cache budget (MiB)")
        ax.set_ylabel("Miss rate")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cache_policy_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    transitions = list(dict.fromkeys(
        (
            row["previous_prompt_category"],
            row["current_prompt_category"],
        )
        for row in transition_rows
    ))
    for phase in ("prefill", "decode"):
        for previous, current in transitions:
            selected = sorted(
                (
                    row for row in transition_rows
                    if row["previous_prompt_category"] == previous
                    and row["current_prompt_category"] == current
                    and row["phase"] == phase
                ),
                key=lambda row: row["layer_index"],
            )
            ax.plot(
                [row["layer_index"] for row in selected],
                [row["unique_set_jaccard"] for row in selected],
                marker=".",
                label=f"{previous}→{current} {phase}",
            )
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("Unique expert-set Jaccard overlap")
    ax.set_title("Routing overlap across adjacent prompt categories")
    ax.set_xticks(range(20))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prompt_transition_overlap.png", dpi=160)
    plt.close(fig)


def main() -> None:
    with REPORT_PATH.open(encoding="utf-8") as handle:
        pilot_report = json.load(handle)
    if pilot_report.get("status") != "complete":
        raise RuntimeError(
            f"Day 6 pilot is not complete: {pilot_report.get('status')}"
        )
    config = load_config()
    rows = read_trace()
    validation = validate_day6_trace(rows, config)
    (
        metric_rows,
        frequency_rows,
        reuse_rows,
        reuse_distribution_rows,
    ) = analyze_prompt_routing(rows, config)
    transitions = prompt_transition_overlap(rows)
    storage_summary, _ = expert_storage_from_safetensors(
        ROOT / "real_trace" / "model_source" / "model.safetensors",
        config,
    )
    cache_rows, layer_cache_rows = replay_scenarios(
        rows,
        storage_summary["expert_weight_bytes_per_layer_expert"],
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        RESULTS_DIR / "expert_usage_frequency.csv",
        [
            "prompt_category", "phase", "layer_index", "expert_id",
            "selection_count", "selection_fraction",
            "phase_total_selections",
        ],
        frequency_rows,
    )
    write_csv(
        RESULTS_DIR / "per_layer_routing_metrics.csv",
        [
            "prompt_category", "phase", "layer_index", "token_count",
            "selection_count", "unique_experts", "top1_share", "top4_share",
            "hhi_concentration", "shannon_entropy_bits",
            "effective_experts_2_pow_entropy", "reuse_count",
            "reuse_distance_mean_requests", "reuse_distance_median_requests",
            "reuse_distance_max_requests", "consecutive_token_pairs",
            "mean_consecutive_overlap_experts",
            "mean_consecutive_token_jaccard",
        ],
        metric_rows,
    )
    write_csv(
        RESULTS_DIR / "reuse_distance_events.csv",
        [
            "prompt_category", "phase", "layer_index", "expert_id",
            "reuse_distance_requests", "previous_request_order",
            "request_order",
        ],
        reuse_rows,
    )
    write_csv(
        RESULTS_DIR / "reuse_distance_distribution.csv",
        [
            "prompt_category", "phase", "layer_aggregation",
            "reuse_distance_requests", "reuse_count",
        ],
        reuse_distribution_rows,
    )
    write_csv(
        RESULTS_DIR / "prompt_transition_overlap.csv",
        [
            "previous_prompt_category", "current_prompt_category", "phase",
            "layer_index", "previous_selection_count",
            "current_selection_count", "previous_unique_experts",
            "current_unique_experts", "shared_experts",
            "unique_set_jaccard", "frequency_cosine_similarity",
        ],
        transitions,
    )
    write_csv(
        RESULTS_DIR / "cache_replay.csv",
        [
            "scenario", "cache_budget_mib", "policy", "prompt_category",
            "phase", "requests", "hits", "misses", "miss_rate", "evictions",
            "expert_weight_bytes_per_entry", "cache_capacity_experts",
            "cache_occupancy_bytes_at_boundary",
            "modeled_expert_weight_bytes_on_miss",
            "traffic_is_modeled_not_measured",
        ],
        cache_rows,
    )
    write_csv(
        RESULTS_DIR / "cache_replay_per_layer.csv",
        [
            "scenario", "cache_budget_mib", "policy", "prompt_category",
            "phase", "layer_index", "requests", "hits", "misses",
            "miss_rate", "evictions_on_layer_requests",
            "modeled_expert_weight_bytes_on_miss",
        ],
        layer_cache_rows,
    )

    aggregate_cache = [
        row for row in cache_rows
        if row["prompt_category"] == "ALL_PROMPTS" and row["phase"] == "all"
    ]
    summary = {
        "status": "complete",
        "trace_validation": validation,
        "pilot": {
            "model_id": pilot_report["model_id"],
            "revision": pilot_report["revision"],
            "model_config": pilot_report["model_config"],
            "generation": pilot_report["generation"],
            "completed_requests": pilot_report["completed_requests"],
            "total_inference_seconds": sum(
                row["inference_seconds"]
                for row in pilot_report["completed_requests"]
            ),
            "cuda_peak_allocated_bytes": (
                pilot_report["cuda_peak_allocated_bytes"]
            ),
            "cuda_peak_reserved_bytes": (
                pilot_report["cuda_peak_reserved_bytes"]
            ),
        },
        "expert_storage": storage_summary,
        "cache_replay": aggregate_cache,
        "limitations": [
            "Four prompts are a small sample and do not establish population-level routing behavior.",
            "Prompt lengths differ; per-prompt selection counts are not directly length-normalized.",
            "Prefill routes all prompt tokens in a batched forward, but replay serializes layer/token/rank events; this is an ordering approximation and does not model coalesced or concurrent expert loads.",
            "Decode routes one input token per forward, but replay remains a cache simulation.",
            "Miss bytes are modeled from expert tensor sizes; no storage or memory traffic was measured.",
            "Cache replay is not a measurement of inference latency or tokens per second.",
            "The tokenizer emitted a warning about its regex pattern and possible incorrect tokenization; the captured routes reflect the tokenizer behavior used in this run. No tokenizer fix or inference retry was applied.",
            "Transformers also emitted a torch_dtype deprecation warning; it did not prevent model loading or trace capture.",
        ],
        "cache_replay_assumptions": {
            "budgets_mib": list(CACHE_BUDGETS_MIB),
            "policies": ["NO_CACHE", "LRU", "LFU"],
            "cache_key": "layer_index:expert_id",
            "reset_scenario": (
                "A fresh cache for each prompt; cache is retained from prefill "
                "through decode within that prompt."
            ),
            "persistent_scenario": (
                "One empty cache persists across all prompts in fixed order."
            ),
            "modeled_transfer_only": True,
        },
    }
    (RESULTS_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    make_figures(frequency_rows, cache_rows, transitions)

    print(
        f"Validated {validation['event_count']} selections across "
        f"{validation['request_count']} prompt requests."
    )
    print("All-prompt cache replay (modeled bytes; not measured traffic):")
    for scenario in ("reset_each_prompt", "persistent_across_prompts"):
        print(scenario)
        for budget in CACHE_BUDGETS_MIB:
            for policy in ("NO_CACHE", "LRU", "LFU"):
                row = next(
                    item for item in aggregate_cache
                    if item["scenario"] == scenario
                    and item["cache_budget_mib"] == budget
                    and item["policy"] == policy
                )
                print(
                    f"  {budget:>3} MiB {policy:<8} "
                    f"hits={row['hits']:>4} misses={row['misses']:>4} "
                    f"miss_rate={row['miss_rate']:.4f} "
                    f"modeled_bytes="
                    f"{row['modeled_expert_weight_bytes_on_miss']}"
                )
    print("Generated Day 6 analysis under results/day6/ and figures/day6/.")


if __name__ == "__main__":
    main()
