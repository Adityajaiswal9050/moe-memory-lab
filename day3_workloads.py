"""Separate synthetic routing workloads for the MoE memory audit.

These workloads do not alter baseline.py or the original baseline trace.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace

from baseline import (
    ACTIVE_EXPERTS_PER_LAYER,
    EXPERTS_PER_LAYER,
    ExpertRequest,
    HOT_SET_INTERVAL,
    NUM_LAYERS,
    SEED,
    TOKEN_STEPS,
)


def _distinct_weighted_experts(rng, experts, weights, count):
    chosen = []
    remaining = list(experts)
    remaining_weights = list(weights)
    for _ in range(count):
        total = sum(remaining_weights)
        pick = rng.random() * total
        for index, weight in enumerate(remaining_weights):
            pick -= weight
            if pick <= 0:
                chosen.append(remaining[index])
                del remaining[index]
                del remaining_weights[index]
                break
        if len(chosen) < len({*chosen}):
            continue
    return tuple(sorted(set(chosen)))


def _sample_two(rng, hot_set, cold_set, hot_probability):
    chosen = []
    for _ in range(ACTIVE_EXPERTS_PER_LAYER):
        pool = hot_set if rng.random() < hot_probability else cold_set
        choices = [expert for expert in pool if expert not in chosen]
        if not choices:
            choices = [expert for expert in range(EXPERTS_PER_LAYER)
                       if expert not in chosen]
        chosen.append(rng.choice(choices))
    return tuple(sorted(set(chosen)))


def static_skewed(seed=SEED):
    rng = random.Random(seed)
    hot_sets = {
        layer: tuple(sorted(rng.sample(range(EXPERTS_PER_LAYER), 4)))
        for layer in range(NUM_LAYERS)
    }
    return _skewed_trace(rng, hot_sets, 0.90, "static_skewed")


def abrupt_changes(seed=SEED):
    rng = random.Random(seed)
    requests = []
    previous = {}
    for step in range(TOKEN_STEPS):
        if step % HOT_SET_INTERVAL == 0:
            hot_sets = {}
            for layer in range(NUM_LAYERS):
                pool = list(range(EXPERTS_PER_LAYER))
                if layer in previous:
                    prior = set(previous[layer])
                    pool = [expert for expert in pool if expert not in prior]
                    if len(pool) < 4:
                        pool = list(range(EXPERTS_PER_LAYER))
                hot_sets[layer] = tuple(sorted(rng.sample(pool, 4)))
            previous = hot_sets
        for layer in range(NUM_LAYERS):
            hot_set = previous[layer]
            cold_set = tuple(
                expert for expert in range(EXPERTS_PER_LAYER)
                if expert not in hot_set
            )
            experts = _sample_two(rng, hot_set, cold_set, 0.90)
            for expert in experts:
                requests.append(ExpertRequest(
                    step, layer, expert, step // HOT_SET_INTERVAL
                ))
    return requests


def gradual_drift(seed=SEED):
    rng = random.Random(seed)
    requests = []
    hot_count = 8
    for step in range(TOKEN_STEPS):
        center_float = (step / (TOKEN_STEPS - 1)) * (EXPERTS_PER_LAYER - 1)
        phase = step // HOT_SET_INTERVAL
        for layer in range(NUM_LAYERS):
            center = (center_float + layer * 3) % EXPERTS_PER_LAYER
            weighted = []
            for expert in range(EXPERTS_PER_LAYER):
                distance = (expert - center) % EXPERTS_PER_LAYER
                distance = min(distance, EXPERTS_PER_LAYER - distance)
                weighted.append((expert, (32 - distance) ** 2))
            weighted.sort(key=lambda item: item[1], reverse=True)
            hot_set = tuple(sorted(expert for expert, _ in weighted[:hot_count]))
            cold_set = tuple(
                expert for expert in range(EXPERTS_PER_LAYER)
                if expert not in hot_set
            )
            hot_weights = [weighted_by_expert for weighted_by_expert in
                           dict(weighted).values() if weighted_by_expert]
            experts = _sample_two(rng, hot_set, cold_set, 0.85)
            for expert in experts:
                requests.append(ExpertRequest(step, layer, expert, phase))
    return requests


def mixed_alternating_tasks(seed=SEED):
    rng = random.Random(seed)
    task_sets = []
    for task in (0, 1):
        task_sets.append({
            layer: tuple(sorted(rng.sample(range(EXPERTS_PER_LAYER), 8)))
            for layer in range(NUM_LAYERS)
        })
    requests = []
    for step in range(TOKEN_STEPS):
        task = (step // 100) % 2
        phase = step // HOT_SET_INTERVAL
        for layer in range(NUM_LAYERS):
            hot_set = task_sets[task][layer]
            cold_set = tuple(
                expert for expert in range(EXPERTS_PER_LAYER)
                if expert not in hot_set
            )
            experts = _sample_two(rng, hot_set, cold_set, 0.85)
            for expert in experts:
                requests.append(ExpertRequest(step, layer, expert, phase))
    return requests


def _skewed_trace(rng, hot_sets, hot_probability, workload_name):
    requests = []
    for step in range(TOKEN_STEPS):
        phase = step // HOT_SET_INTERVAL
        for layer in range(NUM_LAYERS):
            hot_set = hot_sets[layer]
            cold_set = tuple(
                expert for expert in range(EXPERTS_PER_LAYER)
                if expert not in hot_set
            )
            experts = _sample_two(rng, hot_set, cold_set, hot_probability)
            for expert in experts:
                requests.append(ExpertRequest(step, layer, expert, phase))
    del workload_name
    return requests
