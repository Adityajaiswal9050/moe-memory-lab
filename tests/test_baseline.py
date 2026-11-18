import unittest

from baseline import (
    EXPERT_SIZE_MIB,
    LFUCache,
    LRUCache,
    NoCache,
    generate_trace,
    replay,
)


class BaselineTests(unittest.TestCase):
    def test_trace_is_deterministic_and_structurally_valid(self) -> None:
        first = generate_trace()
        second = generate_trace()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1200 * 6 * 2)

        for step in range(1200):
            step_requests = [request for request in first if request.token_step == step]
            self.assertEqual(len(step_requests), 12)
            for layer in range(6):
                experts = [
                    request.expert
                    for request in step_requests
                    if request.layer == layer
                ]
                self.assertEqual(len(experts), 2)
                self.assertEqual(len(set(experts)), 2)

    def test_no_cache_always_misses(self) -> None:
        trace = generate_trace()
        result = replay(trace, NoCache(128))
        self.assertEqual(result["hits"], 0)
        self.assertEqual(result["misses"], len(trace))
        self.assertEqual(result["hit_rate"], 0.0)
        self.assertEqual(result["miss_rate"], 1.0)

    def test_lru_eviction_correctness(self) -> None:
        cache = LRUCache(32)
        self.assertFalse(cache.request("A"))
        self.assertFalse(cache.request("B"))
        self.assertTrue(cache.request("A"))
        self.assertFalse(cache.request("C"))
        self.assertNotIn("B", cache.entries)
        self.assertIn("A", cache.entries)
        self.assertIn("C", cache.entries)

    def test_lfu_frequency_then_recency_tiebreak(self) -> None:
        cache = LFUCache(32)
        self.assertFalse(cache.request("A"))
        self.assertFalse(cache.request("B"))
        self.assertTrue(cache.request("A"))
        self.assertFalse(cache.request("C"))
        self.assertNotIn("B", cache.entries)
        self.assertEqual(cache.entries["A"], 2)

        cache = LFUCache(32)
        cache.request("A")
        cache.request("B")
        cache.request("C")
        self.assertNotIn("A", cache.entries)
        self.assertIn("B", cache.entries)
        self.assertIn("C", cache.entries)

    def test_all_policies_respect_memory_budget(self) -> None:
        trace = generate_trace()
        for budget in (128, 256, 512):
            for cache_type in (LRUCache, LFUCache):
                cache = cache_type(budget)
                replay(trace, cache)
                self.assertLessEqual(cache.used_mib, budget)
                self.assertEqual(cache.used_mib % EXPERT_SIZE_MIB, 0)

    def test_hits_plus_misses_equals_requests(self) -> None:
        trace = generate_trace()
        for budget in (128, 256, 512):
            for cache_type in (NoCache, LRUCache, LFUCache):
                result = replay(trace, cache_type(budget))
                self.assertEqual(
                    result["hits"] + result["misses"],
                    result["requests"],
                )
                self.assertEqual(result["requests"], len(trace))


if __name__ == "__main__":
    unittest.main()
