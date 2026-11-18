"""Analyze and replay the single real MobileMoE smoke-test routing trace."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from safetensors import safe_open

from baseline import LFUCache, LRUCache, NoCache


ROOT = Path(__file__).resolve().parent
TRACE_PATH = ROOT / "real_trace" / "first_real_trace.csv"
CONFIG_PATH = ROOT / "real_trace" / "model_source" / "config.json"
WEIGHTS_PATH = ROOT / "real_trace" / "model_source" / "model.safetensors"
RESULTS_DIR = ROOT / "results" / "day5"
FIGURES_DIR = ROOT / "figures" / "day5"
CACHE_BUDGETS_MIB = (16, 32, 64, 128, 256, 512)
MIB_BYTES = 1024 * 1024
EXPERT_DTYPES_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "BOOL": 1,
}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_trace() -> list[dict]:
    with TRACE_PATH.open(newline="", encoding="utf-8") as handle:
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


def moe_layer_ids(config: dict) -> list[int]:
    count = int(config["num_hidden_layers"])
    step = int(config.get("interleave_moe_layer_step", 1))
    if step < 1:
        raise ValueError("interleave_moe_layer_step must be positive")
    return list(range(step - 1, count, step))


def calculate_expert_weight_bytes(
    expert_tensors: dict[str, dict],
    num_experts: int,
) -> int:
    total_bytes = 0
    for metadata in expert_tensors.values():
        shape = metadata["shape"]
        dtype = metadata["dtype"]
        if not shape or shape[0] != num_experts:
            raise ValueError(f"Unexpected expert tensor shape: {shape}")
        if dtype not in EXPERT_DTYPES_BYTES:
            raise ValueError(f"Unrecognized expert tensor dtype: {dtype}")
        total_bytes += math.prod(shape[1:]) * EXPERT_DTYPES_BYTES[dtype]
    return total_bytes


def validate_trace(rows: list[dict], config: dict) -> dict:
    if not rows:
        raise ValueError("The real trace contains no routing events")
    layers = set(moe_layer_ids(config))
    num_experts = int(config["num_local_experts"])
    top_k = int(config["num_experts_per_tok"])
    if [row["request_order"] for row in rows] != list(range(len(rows))):
        raise ValueError("request_order is not contiguous and chronological")

    phase_order = list(dict.fromkeys(row["phase"] for row in rows))
    if phase_order != ["prefill", "decode"]:
        raise ValueError(f"Unexpected prefill/decode order: {phase_order}")
    request_ids = {row["request_id"] for row in rows}
    if len(request_ids) != 1:
        raise ValueError(f"Expected one request_id, found {sorted(request_ids)}")

    forward_phase: dict[int, str] = {}
    grouped: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    order_keys = []
    for row in rows:
        layer = row["layer_index"]
        expert = row["expert_id"]
        rank = row["selected_rank"]
        if layer not in layers:
            raise ValueError(f"Unexpected MoE layer index: {layer}")
        if not 0 <= expert < num_experts:
            raise ValueError(f"Expert ID out of range: {expert}")
        if not 0 <= rank < top_k:
            raise ValueError(f"Selected rank out of range: {rank}")
        phase = row["phase"]
        if phase not in ("prefill", "decode"):
            raise ValueError(f"Unknown execution phase: {phase}")
        previous_phase = forward_phase.setdefault(row["forward_index"], phase)
        if previous_phase != phase:
            raise ValueError("A model forward contains multiple trace phases")
        grouped[
            (row["forward_index"], layer, row["token_index"])
        ].append(row)
        order_keys.append(
            (
                row["forward_index"],
                layer,
                row["token_index"],
                rank,
            )
        )

    if order_keys != sorted(order_keys):
        raise ValueError(
            "Rows do not preserve forward/layer/token/selected-rank order"
        )
    for key, events in grouped.items():
        ids = [row["expert_id"] for row in events]
        ranks = sorted(row["selected_rank"] for row in events)
        if len(events) != top_k or len(set(ids)) != top_k:
            raise ValueError(f"{key} does not contain {top_k} unique experts")
        if ranks != list(range(top_k)):
            raise ValueError(f"{key} has incomplete selected ranks")

    forward_ids = sorted(forward_phase)
    if forward_ids != list(range(len(forward_ids))):
        raise ValueError("forward_index values are not contiguous")
    for forward_id in forward_ids:
        expected_phase = "prefill" if forward_id == 0 else "decode"
        if forward_phase[forward_id] != expected_phase:
            raise ValueError(
                f"Forward {forward_id} should be {expected_phase}, got "
                f"{forward_phase[forward_id]}"
            )

    token_keys: dict[tuple[int, int], set[int]] = defaultdict(set)
    for forward_id, layer, token_index in grouped:
        token_keys[(forward_id, token_index)].add(layer)
    if any(actual_layers != layers for actual_layers in token_keys.values()):
        raise ValueError("At least one routed token is missing MoE layer events")

    prefill_token_indices = sorted({
        row["token_index"] for row in rows if row["phase"] == "prefill"
    })
    decode_phase_indices = sorted({
        row["phase_token_index"] for row in rows if row["phase"] == "decode"
    })
    prompt_length = len(prefill_token_indices)
    if prefill_token_indices != list(range(prompt_length)):
        raise ValueError("Prefill token indexes are not contiguous from zero")
    if decode_phase_indices != list(range(len(decode_phase_indices))):
        raise ValueError("Decode phase token indexes are not contiguous")
    if any(
        row["token_index"] != prompt_length + row["phase_token_index"]
        for row in rows
        if row["phase"] == "decode"
    ):
        raise ValueError("Decode token indexes do not follow the prompt")

    return {
        "request_id": next(iter(request_ids)),
        "event_count": len(rows),
        "forward_count": len(forward_ids),
        "layer_count": len(layers),
        "layer_indices": sorted(layers),
        "num_experts": num_experts,
        "top_k": top_k,
        "prefill_tokens": len(prefill_token_indices),
        "decode_input_tokens": len(decode_phase_indices),
        "prefill_events": sum(row["phase"] == "prefill" for row in rows),
        "decode_events": sum(row["phase"] == "decode" for row in rows),
        "chronological_order_valid": True,
        "layer_mapping_valid": True,
        "top_k_valid": True,
    }


def expert_storage_from_safetensors(
    weights_path: Path,
    config: dict,
) -> tuple[dict, dict[int, dict]]:
    num_experts = int(config["num_local_experts"])
    expected_layers = set(moe_layer_ids(config))
    tensors: dict[int, dict[str, dict]] = defaultdict(dict)
    pattern = re.compile(
        r"^model\.layers\.(\d+)\.feed_forward\.experts\."
        r"(gate_up_proj|down_proj)$"
    )
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            match = pattern.match(key)
            if not match:
                continue
            layer = int(match.group(1))
            param = match.group(2)
            tensor_slice = handle.get_slice(key)
            tensors[layer][param] = {
                "shape": tensor_slice.get_shape(),
                "dtype": tensor_slice.get_dtype(),
            }

    if set(tensors) != expected_layers:
        raise ValueError(
            f"Safetensors expert layers {sorted(tensors)} do not match "
            f"configured MoE layers {sorted(expected_layers)}"
        )

    per_layer = {}
    all_sizes = set()
    all_dtypes = set()
    for layer in sorted(expected_layers):
        layer_tensors = tensors[layer]
        if set(layer_tensors) != {"gate_up_proj", "down_proj"}:
            raise ValueError(f"Layer {layer} is missing expert weight tensors")
        total_bytes_per_expert = calculate_expert_weight_bytes(
            layer_tensors, num_experts
        )
        descriptions = {}
        for name, metadata in layer_tensors.items():
            shape = metadata["shape"]
            dtype = metadata["dtype"]
            elements_per_expert = math.prod(shape[1:])
            tensor_bytes_per_expert = (
                elements_per_expert * EXPERT_DTYPES_BYTES[dtype]
            )
            descriptions[name] = {
                "shape": shape,
                "dtype": dtype,
                "bytes_per_expert": tensor_bytes_per_expert,
            }
            all_dtypes.add(dtype)
        all_sizes.add(total_bytes_per_expert)
        per_layer[layer] = {
            "expert_tensor_details": descriptions,
            "expert_weight_bytes": total_bytes_per_expert,
        }

    if len(all_sizes) != 1 or len(all_dtypes) != 1:
        raise ValueError("Expert tensors do not have a uniform size and dtype")
    expert_bytes = next(iter(all_sizes))
    dtype = next(iter(all_dtypes))
    summary = {
        "model_id": "facebook/MobileMoE-S-SFT",
        "revision": "4decae01abfeaf46046b8774fdc4404b3e33a3a4",
        "tensor_file": str(weights_path.relative_to(ROOT)),
        "num_moe_layers": len(expected_layers),
        "num_experts_per_layer": num_experts,
        "expert_tensor_dtype": dtype,
        "expert_weight_bytes_per_layer_expert": expert_bytes,
        "expert_weight_mib_per_layer_expert": expert_bytes / MIB_BYTES,
        "all_routed_expert_weight_bytes": (
            expert_bytes * len(expected_layers) * num_experts
        ),
        "included_tensors": ["gate_up_proj", "down_proj"],
        "excluded_weights": [
            "router weights and bias",
            "shared expert weights",
            "attention and embedding weights",
            "allocator and runtime overhead",
        ],
        "basis": (
            "Safetensors tensor shapes and dtype; this is a modeled cache "
            "unit, not measured disk traffic."
        ),
    }
    return summary, per_layer


def phase_layer_events(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    result: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        result[(row["phase"], row["layer_index"])].append(row)
    return result


def analyze_usage(rows: list[dict], config: dict):
    num_experts = int(config["num_local_experts"])
    layer_events = phase_layer_events(rows)
    metric_rows = []
    frequency_rows = []
    reuse_rows = []
    reuse_distribution = Counter()
    frequency_matrices = {}

    for phase in ("prefill", "decode"):
        frequency_matrices[phase] = []
        for layer in range(int(config["num_hidden_layers"])):
            events = sorted(
                layer_events[(phase, layer)],
                key=lambda row: row["request_order"],
            )
            frequencies = Counter(row["expert_id"] for row in events)
            total = len(events)
            probabilities = [
                frequencies[expert] / total for expert in frequencies
            ] if total else []
            hhi = sum(probability ** 2 for probability in probabilities)
            entropy = -sum(
                probability * math.log2(probability)
                for probability in probabilities
            ) if probabilities else 0.0
            top_counts = sorted(frequencies.values(), reverse=True)

            for expert in range(num_experts):
                count = frequencies.get(expert, 0)
                frequency_rows.append({
                    "phase": phase,
                    "layer_index": layer,
                    "expert_id": expert,
                    "selection_count": count,
                    "selection_fraction": count / total if total else 0.0,
                    "phase_total_selections": total,
                })

            token_experts: dict[int, set[int]] = defaultdict(set)
            last_seen: dict[int, int] = {}
            reuse_distances = []
            for position, event in enumerate(events):
                expert = event["expert_id"]
                token_experts[event["token_index"]].add(expert)
                if expert in last_seen:
                    distance = position - last_seen[expert] - 1
                    reuse_distances.append(distance)
                    reuse_distribution[(phase, distance)] += 1
                    reuse_rows.append({
                        "phase": phase,
                        "layer_index": layer,
                        "expert_id": expert,
                        "reuse_distance_requests": distance,
                        "previous_request_order": events[last_seen[expert]][
                            "request_order"
                        ],
                        "request_order": event["request_order"],
                    })
                last_seen[expert] = position

            tokens = sorted(token_experts)
            overlap_counts = []
            jaccards = []
            for previous, current in zip(tokens, tokens[1:]):
                previous_set = token_experts[previous]
                current_set = token_experts[current]
                intersection = len(previous_set & current_set)
                union = len(previous_set | current_set)
                overlap_counts.append(intersection)
                jaccards.append(intersection / union if union else 0.0)

            metric_rows.append({
                "phase": phase,
                "layer_index": layer,
                "token_count": len(tokens),
                "selection_count": total,
                "unique_experts": len(frequencies),
                "top1_share": top_counts[0] / total if total else 0.0,
                "top4_share": sum(top_counts[:4]) / total if total else 0.0,
                "hhi_concentration": hhi,
                "shannon_entropy_bits": entropy,
                "effective_experts_2_pow_entropy": 2 ** entropy,
                "reuse_count": len(reuse_distances),
                "reuse_distance_mean_requests": (
                    sum(reuse_distances) / len(reuse_distances)
                    if reuse_distances else ""
                ),
                "reuse_distance_median_requests": (
                    sorted(reuse_distances)[len(reuse_distances) // 2]
                    if reuse_distances else ""
                ),
                "reuse_distance_max_requests": (
                    max(reuse_distances) if reuse_distances else ""
                ),
                "consecutive_token_pairs": len(overlap_counts),
                "mean_consecutive_overlap_experts": (
                    sum(overlap_counts) / len(overlap_counts)
                    if overlap_counts else ""
                ),
                "mean_consecutive_token_jaccard": (
                    sum(jaccards) / len(jaccards) if jaccards else ""
                ),
            })
            frequency_matrices[phase].append([
                frequencies.get(expert, 0) for expert in range(num_experts)
            ])

    reuse_hist_rows = [
        {
            "phase": phase,
            "layer_aggregation": "pooled_across_layers",
            "reuse_distance_requests": distance,
            "reuse_count": count,
        }
        for (phase, distance), count in sorted(reuse_distribution.items())
    ]
    return metric_rows, frequency_rows, reuse_rows, reuse_hist_rows, frequency_matrices


def make_cache(policy_name: str, budget_bytes: int, expert_bytes: int):
    capacity = budget_bytes // expert_bytes
    if policy_name == "NO_CACHE":
        return NoCache(budget_mib=0, expert_size_mib=1), capacity
    if policy_name == "LRU":
        return LRUCache(budget_mib=capacity, expert_size_mib=1), capacity
    if policy_name == "LFU":
        return LFUCache(budget_mib=capacity, expert_size_mib=1), capacity
    raise ValueError(f"Unknown policy: {policy_name}")


def cache_entry_count(cache) -> int:
    return len(getattr(cache, "entries", {}))


def replay_caches(rows: list[dict], expert_bytes: int):
    summary_rows = []
    layer_rows = []
    policies = ("NO_CACHE", "LRU", "LFU")
    phases = ("prefill", "decode")

    for budget_mib in CACHE_BUDGETS_MIB:
        budget_bytes = budget_mib * MIB_BYTES
        for policy in policies:
            cache, capacity = make_cache(policy, budget_bytes, expert_bytes)
            phase_counts = {
                phase: {"hits": 0, "misses": 0, "evictions": 0}
                for phase in phases
            }
            layer_counts = {
                (phase, layer): {"hits": 0, "misses": 0, "evictions": 0}
                for phase in phases
                for layer in range(20)
            }
            phase_occupancy = {}

            for position, event in enumerate(rows):
                phase = event["phase"]
                layer = event["layer_index"]
                key = f"L{layer}:E{event['expert_id']}"
                entries_before = cache_entry_count(cache)
                hit = cache.request(key)
                count_key = (phase, layer)
                if hit:
                    phase_counts[phase]["hits"] += 1
                    layer_counts[count_key]["hits"] += 1
                else:
                    phase_counts[phase]["misses"] += 1
                    layer_counts[count_key]["misses"] += 1
                    if policy != "NO_CACHE" and entries_before >= capacity:
                        phase_counts[phase]["evictions"] += 1
                        layer_counts[count_key]["evictions"] += 1

                next_phase = (
                    rows[position + 1]["phase"]
                    if position + 1 < len(rows) else None
                )
                if next_phase != phase:
                    phase_occupancy[phase] = cache_entry_count(cache)

            all_counts = {
                "hits": sum(count["hits"] for count in phase_counts.values()),
                "misses": sum(count["misses"] for count in phase_counts.values()),
                "evictions": sum(
                    count["evictions"] for count in phase_counts.values()
                ),
            }
            for phase in ("prefill", "decode", "all"):
                counts = all_counts if phase == "all" else phase_counts[phase]
                requests = counts["hits"] + counts["misses"]
                occupancy = (
                    cache_entry_count(cache)
                    if phase in ("decode", "all")
                    else phase_occupancy.get(phase, 0)
                )
                summary_rows.append({
                    "cache_budget_mib": budget_mib,
                    "cache_budget_bytes": budget_bytes,
                    "policy": policy,
                    "phase": phase,
                    "requests": requests,
                    "hits": counts["hits"],
                    "misses": counts["misses"],
                    "miss_rate": counts["misses"] / requests if requests else 0,
                    "evictions": counts["evictions"],
                    "expert_weight_bytes_per_entry": expert_bytes,
                    "cache_capacity_experts": capacity if policy != "NO_CACHE" else 0,
                    "cache_occupancy_bytes_at_phase_end": (
                        occupancy * expert_bytes
                        if policy != "NO_CACHE" else 0
                    ),
                    "modeled_expert_weight_bytes_on_miss": (
                        counts["misses"] * expert_bytes
                    ),
                    "traffic_is_modeled_not_measured": True,
                })

            for phase in ("prefill", "decode", "all"):
                for layer in range(20):
                    if phase == "all":
                        counts = {
                            key: (
                                layer_counts[("prefill", layer)][key]
                                + layer_counts[("decode", layer)][key]
                            )
                            for key in ("hits", "misses", "evictions")
                        }
                    else:
                        counts = layer_counts[(phase, layer)]
                    requests = counts["hits"] + counts["misses"]
                    layer_rows.append({
                        "cache_budget_mib": budget_mib,
                        "policy": policy,
                        "phase": phase,
                        "layer_index": layer,
                        "requests": requests,
                        "hits": counts["hits"],
                        "misses": counts["misses"],
                        "miss_rate": (
                            counts["misses"] / requests if requests else 0
                        ),
                        "evictions_on_layer_requests": counts["evictions"],
                        "modeled_expert_weight_bytes_on_miss": (
                            counts["misses"] * expert_bytes
                        ),
                    })

    return summary_rows, layer_rows


def make_figures(metric_rows, frequency_matrices, cache_rows, layer_cache_rows,
                 reuse_hist_rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for phase in ("prefill", "decode"):
        matrix = frequency_matrices[phase]
        fig, ax = plt.subplots(figsize=(13, 6))
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
        ax.set_title(f"MobileMoE expert selections by layer — {phase}")
        ax.set_xlabel("Expert ID")
        ax.set_ylabel("MoE layer index")
        ax.set_xticks(range(0, 60, 5))
        ax.set_yticks(range(20))
        fig.colorbar(image, ax=ax, label="Selected expert count")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"expert_usage_heatmap_{phase}.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for phase in ("prefill", "decode"):
        values = [
            row["reuse_distance_requests"]
            for row in reuse_hist_rows if row["phase"] == phase
        ]
        weights = [
            row["reuse_count"]
            for row in reuse_hist_rows if row["phase"] == phase
        ]
        if values:
            axes[0 if phase == "prefill" else 1].bar(
                values, weights, width=1.0, alpha=0.8
            )
        ax = axes[0 if phase == "prefill" else 1]
        ax.set_title(f"Reuse distance pooled across layers — {phase}")
        ax.set_xlabel("Intervening expert requests")
        ax.set_ylabel("Reuse events")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reuse_distance_distribution.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for phase in ("prefill", "decode"):
        phase_rows = [row for row in metric_rows if row["phase"] == phase]
        x = [row["layer_index"] for row in phase_rows]
        axes[0].plot(
            x, [row["top1_share"] for row in phase_rows],
            marker="o", label=phase,
        )
        axes[1].plot(
            x, [row["hhi_concentration"] for row in phase_rows],
            marker="o", label=phase,
        )
    axes[0].set_title("Most-selected expert share")
    axes[0].set_ylabel("Top-1 selection fraction")
    axes[1].set_title("Expert frequency HHI")
    axes[1].set_ylabel("HHI (higher is more concentrated)")
    for ax in axes:
        ax.set_xlabel("MoE layer index")
        ax.set_xticks(range(20))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "routing_frequency_concentration.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for row_index, phase in enumerate(("prefill", "decode")):
        for metric, ax in (
            ("miss_rate", axes[row_index][0]),
            ("modeled_expert_weight_bytes_on_miss", axes[row_index][1]),
        ):
            for policy in ("LRU", "LFU", "NO_CACHE"):
                selected = [
                    row for row in cache_rows
                    if row["phase"] == phase and row["policy"] == policy
                ]
                selected.sort(key=lambda row: row["cache_budget_mib"])
                ax.plot(
                    [row["cache_budget_mib"] for row in selected],
                    [row[metric] for row in selected],
                    marker="o",
                    label=policy,
                )
            ax.set_title(
                f"{phase}: "
                + ("miss rate" if metric == "miss_rate"
                   else "modeled expert-weight bytes")
            )
            ax.set_xlabel("Cache budget (MiB)")
            ax.set_ylabel(
                "Miss fraction" if metric == "miss_rate"
                else "Modeled bytes"
            )
            ax.grid(alpha=0.25)
            ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cache_miss_rate_and_modeled_bytes.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    selected_rows = [
        row for row in layer_cache_rows
        if row["phase"] in ("prefill", "decode")
    ]
    for ax, phase in zip(axes, ("prefill", "decode")):
        policies_budgets = [
            (policy, budget)
            for budget in CACHE_BUDGETS_MIB
            for policy in ("LRU", "LFU", "NO_CACHE")
        ]
        matrix = []
        labels = []
        for policy, budget in policies_budgets:
            by_layer = {
                row["layer_index"]: row["miss_rate"]
                for row in selected_rows
                if row["phase"] == phase
                and row["policy"] == policy
                and row["cache_budget_mib"] == budget
            }
            matrix.append([by_layer.get(layer, 0.0) for layer in range(20)])
            labels.append(f"{budget} MiB {policy}")
        image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_title(f"Per-layer cache miss rate — {phase}")
        ax.set_xlabel("MoE layer index")
        ax.set_ylabel("Budget and policy")
        ax.set_xticks(range(20))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        fig.colorbar(image, ax=ax, label="Miss rate")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_layer_cache_miss_rate.png", dpi=160)
    plt.close(fig)


def main() -> None:
    config = load_config()
    rows = read_trace()
    validation = validate_trace(rows, config)
    storage_summary, per_layer_storage = expert_storage_from_safetensors(
        WEIGHTS_PATH, config
    )
    metric_rows, frequency_rows, reuse_rows, reuse_hist_rows, frequency_matrices = (
        analyze_usage(rows, config)
    )
    cache_rows, layer_cache_rows = replay_caches(
        rows,
        storage_summary["expert_weight_bytes_per_layer_expert"],
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "trace_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "model_expert_storage.json").write_text(
        json.dumps(
            {
                **storage_summary,
                "per_layer": per_layer_storage,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    write_csv(
        RESULTS_DIR / "expert_usage_frequency.csv",
        [
            "phase", "layer_index", "expert_id", "selection_count",
            "selection_fraction", "phase_total_selections",
        ],
        frequency_rows,
    )
    write_csv(
        RESULTS_DIR / "per_layer_routing_metrics.csv",
        [
            "phase", "layer_index", "token_count", "selection_count",
            "unique_experts", "top1_share", "top4_share",
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
            "phase", "layer_index", "expert_id", "reuse_distance_requests",
            "previous_request_order", "request_order",
        ],
        reuse_rows,
    )
    write_csv(
        RESULTS_DIR / "reuse_distance_distribution.csv",
        [
            "phase", "layer_aggregation", "reuse_distance_requests",
            "reuse_count",
        ],
        reuse_hist_rows,
    )
    write_csv(
        RESULTS_DIR / "real_trace_cache_baseline.csv",
        [
            "cache_budget_mib", "cache_budget_bytes", "policy", "phase",
            "requests", "hits", "misses", "miss_rate", "evictions",
            "expert_weight_bytes_per_entry", "cache_capacity_experts",
            "cache_occupancy_bytes_at_phase_end",
            "modeled_expert_weight_bytes_on_miss",
            "traffic_is_modeled_not_measured",
        ],
        cache_rows,
    )
    write_csv(
        RESULTS_DIR / "real_trace_cache_per_layer.csv",
        [
            "cache_budget_mib", "policy", "phase", "layer_index", "requests",
            "hits", "misses", "miss_rate", "evictions_on_layer_requests",
            "modeled_expert_weight_bytes_on_miss",
        ],
        layer_cache_rows,
    )
    make_figures(
        metric_rows,
        frequency_matrices,
        cache_rows,
        layer_cache_rows,
        reuse_hist_rows,
    )

    print(
        f"Trace: {validation['event_count']} selections; "
        f"{validation['prefill_tokens']} prefill tokens, "
        f"{validation['decode_input_tokens']} decode input tokens."
    )
    print(
        f"Expert weights: "
        f"{storage_summary['expert_weight_bytes_per_layer_expert']} bytes/"
        f"expert ({storage_summary['expert_weight_mib_per_layer_expert']:.4f} MiB)."
    )
    print("All-request cache replay:")
    print("Budget MiB  Policy  Hits  Misses  Miss rate  Modeled bytes")
    for budget in CACHE_BUDGETS_MIB:
        for policy in ("NO_CACHE", "LRU", "LFU"):
            row = next(
                item for item in cache_rows
                if item["cache_budget_mib"] == budget
                and item["phase"] == "all"
                and item["policy"] == policy
            )
            print(
                f"{budget:<11} {policy:<7} {row['hits']:<5} "
                f"{row['misses']:<7} {row['miss_rate']:.4f} "
                f"{row['modeled_expert_weight_bytes_on_miss']}"
            )
    print("Generated Day 5 analysis under results/day5/ and figures/day5/.")


if __name__ == "__main__":
    main()
