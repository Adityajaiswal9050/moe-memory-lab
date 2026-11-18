import unittest

from analysis import PHASES, alternate_order, replay_detailed
from baseline import LFUCache, LRUCache, generate_trace


class AnalysisTests(unittest.TestCase):
    def test_phase_boundaries_cover_exact_trace_range(self) -> None:
        self.assertEqual(PHASES, (
            (0, 249),
            (250, 499),
            (500, 749),
            (750, 999),
            (1000, 1199),
        ))
        covered = [
            token
            for start, end in PHASES
            for token in range(start, end + 1)
        ]
        self.assertEqual(covered, list(range(1200)))
        self.assertEqual(len(set(covered)), 1200)

    def test_phase_hits_misses_and_requests_account_for_trace(self) -> None:
        trace = generate_trace(seed=42)
        for cache_type in (LRUCache, LFUCache):
            detailed = replay_detailed(trace, cache_type(256))
            for row in detailed["phase_rows"]:
                self.assertEqual(
                    row["hits"] + row["misses"],
                    row["requests"],
                )
            self.assertEqual(
                sum(row["requests"] for row in detailed["phase_rows"]),
                len(trace),
            )
            self.assertEqual(detailed["hits"] + detailed["misses"], len(trace))

    def test_alternative_order_preserves_same_requests(self) -> None:
        trace = generate_trace(seed=42)
        alternate = alternate_order(trace)
        self.assertEqual(len(alternate), len(trace))
        self.assertCountEqual(
            [(request.token_step, request.key) for request in alternate],
            [(request.token_step, request.key) for request in trace],
        )

    def test_lfu_eviction_trace_is_hand_constructed_and_deterministic(self) -> None:
        cache = LFUCache(32)
        sequence = ["A", "A", "B", "B", "A", "C", "C", "C"]
        results = [cache.request(key) for key in sequence]

        self.assertEqual(
            results,
            [False, True, False, True, True, False, True, True],
        )
        self.assertIn("A", cache.entries)
        self.assertIn("C", cache.entries)
        self.assertNotIn("B", cache.entries)


if __name__ == "__main__":
    unittest.main()
