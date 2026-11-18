import unittest

from day5_analysis import (
    CACHE_BUDGETS_MIB,
    MIB_BYTES,
    analyze_usage,
    calculate_expert_weight_bytes,
    replay_caches,
    validate_trace,
)


CONFIG = {
    "num_hidden_layers": 2,
    "interleave_moe_layer_step": 1,
    "num_local_experts": 3,
    "num_experts_per_tok": 2,
}


def synthetic_trace():
    rows = []
    forwards = (
        (0, "prefill", (0, 1)),
        (1, "decode", (2,)),
    )
    for forward_index, phase, token_indices in forwards:
        for layer_index in range(CONFIG["num_hidden_layers"]):
            for token_index in token_indices:
                phase_token_index = token_index if phase == "prefill" else 0
                first_expert = (token_index + layer_index) % CONFIG[
                    "num_local_experts"
                ]
                experts = (
                    first_expert,
                    (first_expert + 1) % CONFIG["num_local_experts"],
                )
                for selected_rank, expert_id in enumerate(experts):
                    rows.append({
                        "request_order": len(rows),
                        "request_id": "fixture-request",
                        "forward_index": forward_index,
                        "phase": phase,
                        "phase_token_index": phase_token_index,
                        "token_index": token_index,
                        "layer_index": layer_index,
                        "expert_id": expert_id,
                        "selected_rank": selected_rank,
                    })
    return rows


class Day5AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_trace()
        self.config = dict(CONFIG)

    def test_trace_validation_and_phase_separation(self):
        validation = validate_trace(self.rows, self.config)
        self.assertTrue(validation["chronological_order_valid"])
        self.assertEqual(validation["layer_indices"], [0, 1])
        self.assertEqual(validation["prefill_events"], 8)
        self.assertEqual(validation["decode_events"], 4)
        self.assertEqual(validation["prefill_tokens"], 2)
        self.assertEqual(validation["decode_input_tokens"], 1)

    def test_invalid_synthetic_trace_is_rejected(self):
        broken_rows = [dict(row) for row in self.rows]
        broken_rows[1]["expert_id"] = broken_rows[0]["expert_id"]
        with self.assertRaisesRegex(ValueError, "unique experts"):
            validate_trace(broken_rows, self.config)

    def test_expert_size_calculation_uses_fixture_shapes_and_dtypes(self):
        metadata = {
            "gate_up_proj": {"shape": (3, 2, 4), "dtype": "BF16"},
            "down_proj": {"shape": (3, 1, 4), "dtype": "F32"},
        }
        self.assertEqual(calculate_expert_weight_bytes(metadata, 3), 32)

    def test_usage_metrics_and_cache_accounting_use_only_fixture_data(self):
        metrics, frequencies, _, _, _ = analyze_usage(
            self.rows,
            self.config,
        )
        self.assertEqual(len(metrics), 4)
        self.assertEqual(len(frequencies), 2 * 2 * 3)
        self.assertTrue(all(row["unique_experts"] <= 3 for row in metrics))

        expert_bytes = 32
        cache_rows, layer_rows = replay_caches(self.rows, expert_bytes)
        all_rows = [row for row in cache_rows if row["phase"] == "all"]
        self.assertEqual(len(all_rows), len(CACHE_BUDGETS_MIB) * 3)
        for row in cache_rows:
            self.assertEqual(row["hits"] + row["misses"], row["requests"])
            self.assertEqual(
                row["modeled_expert_weight_bytes_on_miss"],
                row["misses"] * expert_bytes,
            )
            self.assertLessEqual(
                row["cache_occupancy_bytes_at_phase_end"],
                row["cache_budget_mib"] * MIB_BYTES,
            )
        self.assertEqual(
            next(
                row["misses"] for row in all_rows
                if row["policy"] == "NO_CACHE"
            ),
            len(self.rows),
        )
        self.assertEqual(
            len(layer_rows),
            len(CACHE_BUDGETS_MIB) * 3 * 3 * 20,
        )


if __name__ == "__main__":
    unittest.main()
