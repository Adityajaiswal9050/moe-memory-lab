import unittest

from day6_analysis import (
    analyze_prompt_routing,
    prompt_transition_overlap,
    replay_scenarios,
    validate_day6_trace,
)


class Day6AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {
            "num_hidden_layers": 1,
            "interleave_moe_layer_step": 1,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        }
        cls.rows = []
        order = 0
        categories = (
            "coding",
            "mathematics",
            "general_knowledge",
            "scientific_reasoning",
        )
        for request_index, category in enumerate(categories):
            request_id = f"request-{request_index}"
            token_routes = (
                ("prefill", 0, 0, (1, 2)),
                ("prefill", 1, 1, (1, 3)),
                ("decode", 0, 2, (1, 0)),
            )
            for phase, phase_token, token_index, experts in token_routes:
                forward_index = 0 if phase == "prefill" else 1
                for rank, expert in enumerate(experts):
                    cls.rows.append({
                        "request_order": order,
                        "request_id": request_id,
                        "forward_index": forward_index,
                        "phase": phase,
                        "phase_token_index": phase_token,
                        "token_index": token_index,
                        "layer_index": 0,
                        "expert_id": expert,
                        "selected_rank": rank,
                        "prompt_category": category,
                        "seed": 42,
                        "max_new_tokens": 8,
                        "do_sample": False,
                        "temperature": 1.0,
                        "top_p": 1.0,
                    })
                    order += 1

    def test_trace_order_top_k_layers_and_request_boundaries(self):
        summary = validate_day6_trace(self.rows, self.config)
        self.assertTrue(summary["global_request_order_valid"])
        self.assertTrue(summary["per_request_top_k_and_layer_mapping_valid"])
        self.assertEqual(summary["request_count"], 4)
        self.assertEqual(summary["event_count"], 24)

    def test_prompt_frequency_metrics_and_transition_overlap(self):
        metrics, frequencies, reuse, reuse_distribution = (
            analyze_prompt_routing(self.rows, self.config)
        )
        self.assertEqual(len(metrics), 8)
        self.assertEqual(len(frequencies), 4 * 2 * 1 * 4)
        self.assertTrue(all(
            row["prompt_category"] for row in metrics + frequencies
        ))
        self.assertTrue(reuse)
        self.assertTrue(reuse_distribution)
        transitions = prompt_transition_overlap(self.rows)
        self.assertEqual(len(transitions), 3 * 2)
        self.assertTrue(all(
            row["unique_set_jaccard"] == 1.0 for row in transitions
        ))
        self.assertTrue(all(
            row["frequency_cosine_similarity"] > 0.99
            for row in transitions
        ))

    def test_cache_reset_and_persistent_scenarios_are_distinct(self):
        summary, per_layer = replay_scenarios(
            self.rows,
            expert_bytes=1024,
            budgets=(16,),
        )
        persistent = next(
            row for row in summary
            if row["scenario"] == "persistent_across_prompts"
            and row["policy"] == "LRU"
            and row["prompt_category"] == "ALL_PROMPTS"
        )
        reset = next(
            row for row in summary
            if row["scenario"] == "reset_each_prompt"
            and row["policy"] == "LRU"
            and row["prompt_category"] == "ALL_PROMPTS"
        )
        self.assertEqual(
            (persistent["hits"], persistent["misses"]),
            (20, 4),
        )
        self.assertEqual((reset["hits"], reset["misses"]), (8, 16))
        for row in summary:
            self.assertEqual(
                row["hits"] + row["misses"],
                row["requests"],
            )
            self.assertEqual(
                row["modeled_expert_weight_bytes_on_miss"],
                row["misses"] * 1024,
            )
        persistent_second_prompt = next(
            row for row in summary
            if row["scenario"] == "persistent_across_prompts"
            and row["policy"] == "LRU"
            and row["prompt_category"] == "mathematics"
            and row["phase"] == "all"
        )
        reset_second_prompt = next(
            row for row in summary
            if row["scenario"] == "reset_each_prompt"
            and row["policy"] == "LRU"
            and row["prompt_category"] == "mathematics"
            and row["phase"] == "all"
        )
        self.assertEqual(persistent_second_prompt["misses"], 0)
        self.assertEqual(reset_second_prompt["misses"], 4)
        self.assertTrue(per_layer)

    def test_no_cache_always_misses_and_cache_occupancy_stays_bounded(self):
        summary, _ = replay_scenarios(
            self.rows,
            expert_bytes=1024,
            budgets=(16,),
        )
        no_cache_rows = [
            row for row in summary if row["policy"] == "NO_CACHE"
        ]
        self.assertTrue(all(
            row["hits"] == 0 and row["misses"] == row["requests"]
            for row in no_cache_rows
        ))
        for row in summary:
            occupancy = row["cache_occupancy_bytes_at_boundary"]
            if occupancy != "":
                self.assertLessEqual(
                    occupancy,
                    row["cache_budget_mib"] * 1024 * 1024,
                )


if __name__ == "__main__":
    unittest.main()
