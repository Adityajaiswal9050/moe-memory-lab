import unittest

from day7_analysis import replay_sequence
from day7_policies import DecayedLFUCache, SegmentedLRUCache


class Day7PolicyTests(unittest.TestCase):
    def test_decayed_lfu_uses_lazy_exponential_aging(self):
        cache = DecayedLFUCache(capacity_experts=2, decay_alpha=0.5)
        self.assertFalse(cache.request("A"))
        self.assertFalse(cache.request("B"))
        self.assertTrue(cache.request("A"))
        self.assertAlmostEqual(cache.frequency("A"), 1.25)
        self.assertAlmostEqual(cache.frequency("B"), 0.5)
        self.assertFalse(cache.request("C"))
        self.assertNotIn("B", cache.entries)
        self.assertIn("A", cache.entries)
        self.assertIn("C", cache.entries)
        self.assertEqual(cache.evictions, 1)

    def test_decayed_lfu_recency_breaks_equal_frequency_ties(self):
        cache = DecayedLFUCache(capacity_experts=2, decay_alpha=1.0)
        cache.request("older")
        cache.request("newer")
        cache.request("third")
        self.assertNotIn("older", cache.entries)
        self.assertIn("newer", cache.entries)
        self.assertIn("third", cache.entries)

    def test_decayed_lfu_rejects_invalid_coefficients_and_capacity(self):
        for alpha in (0.0, -0.1, 1.01):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    DecayedLFUCache(2, alpha)
        with self.assertRaises(ValueError):
            DecayedLFUCache(-1, 0.9)

    def test_slru_promotes_reused_entries_and_evicts_probationary_lru(self):
        cache = SegmentedLRUCache(
            capacity_experts=2,
            probationary_ratio=0.5,
        )
        self.assertFalse(cache.request("A"))
        self.assertTrue(cache.request("A"))
        self.assertIn("A", cache.protected)
        self.assertFalse(cache.request("B"))
        self.assertTrue(cache.request("B"))
        self.assertIn("B", cache.protected)
        self.assertIn("A", cache.probationary)
        self.assertFalse(cache.request("C"))
        self.assertNotIn("A", cache.probationary)
        self.assertIn("B", cache.protected)
        self.assertIn("C", cache.probationary)
        self.assertEqual(cache.used_experts, 2)
        self.assertEqual(cache.evictions, 1)

    def test_slru_handles_single_entry_and_zero_capacity(self):
        one = SegmentedLRUCache(1)
        self.assertFalse(one.request("A"))
        self.assertTrue(one.request("A"))
        self.assertFalse(one.request("B"))
        self.assertEqual(one.used_experts, 1)

        empty = SegmentedLRUCache(0)
        self.assertFalse(empty.request("A"))
        self.assertEqual(empty.used_experts, 0)

    def test_replay_accounting_budget_and_reproducibility(self):
        events = [
            {"request_order": i, "key": key, "phase": phase,
             "layer_index": layer}
            for i, (key, phase, layer) in enumerate((
                ("L0:E1", "prefill", 0),
                ("L0:E1", "prefill", 0),
                ("L1:E2", "prefill", 1),
                ("L0:E1", "decode", 0),
                ("L1:E3", "decode", 1),
                ("L1:E2", "decode", 1),
            ))
        ]
        results = []
        for _ in range(2):
            cache = DecayedLFUCache(capacity_experts=2, decay_alpha=0.95)
            summary, _ = replay_sequence(
                events,
                cache,
                "DECAYED_LFU",
                expert_bytes=16 * 1024 * 1024,
                budget_mib=32,
                alpha=0.95,
                scenario="cold_start",
                workload="test",
                seed=42,
                prompt_category="unit",
            )
            all_requests = next(row for row in summary if row["phase"] == "all")
            self.assertEqual(
                all_requests["hits"] + all_requests["misses"],
                len(events),
            )
            self.assertEqual(
                all_requests["modeled_expert_bytes_transferred"],
                all_requests["misses"] * 16 * 1024 * 1024,
            )
            self.assertLessEqual(
                all_requests["cache_occupancy_bytes_at_boundary"],
                32 * 1024 * 1024,
            )
            self.assertGreater(
                all_requests["estimated_policy_metadata_bytes"],
                0,
            )
            results.append((
                all_requests["hits"],
                all_requests["misses"],
                all_requests["evictions"],
            ))
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
