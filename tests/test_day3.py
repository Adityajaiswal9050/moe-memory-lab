import unittest
from collections import Counter

from baseline import generate_trace
from day3_workloads import (
    abrupt_changes,
    gradual_drift,
    mixed_alternating_tasks,
    static_skewed,
)


class Day3TraceTests(unittest.TestCase):
    def test_original_baseline_has_no_cold_requests_and_is_24_expert_per_phase(self):
        trace = generate_trace(seed=42)
        for start, end in (
            (0, 249),
            (250, 499),
            (500, 749),
            (750, 999),
            (1000, 1199),
        ):
            phase = [
                request for request in trace
                if start <= request.token_step <= end
            ]
            self.assertEqual(len({request.key for request in phase}), 24)

    def test_original_baseline_frequency_histogram_only_hot_experts(self):
        trace = generate_trace(seed=42)
        histogram = Counter(request.key for request in trace)
        self.assertGreater(len(histogram), 24)

    def test_new_workloads_are_deterministic(self):
        workloads = (
            static_skewed,
            abrupt_changes,
            gradual_drift,
            mixed_alternating_tasks,
        )
        for workload in workloads:
            self.assertEqual(workload(seed=42), workload(seed=42))

    def test_new_workload_validity_per_token_and_per_layer(self):
        for workload in (
            static_skewed,
            abrupt_changes,
            gradual_drift,
            mixed_alternating_tasks,
        ):
            trace = workload(seed=42)
            self.assertEqual(len(trace), 1200 * 6 * 2)
            for step in range(1200):
                step_requests = [
                    request for request in trace
                    if request.token_step == step
                ]
                self.assertEqual(len(step_requests), 12)
                for layer in range(6):
                    experts = [
                        request.expert for request in step_requests
                        if request.layer == layer
                    ]
                    self.assertEqual(len(experts), 2)
                    self.assertEqual(len(set(experts)), 2)
                    self.assertTrue(all(0 <= expert < 32 for expert in experts))

    def test_new_workloads_include_real_cold_tail(self):
        for workload in (
            static_skewed,
            abrupt_changes,
            gradual_drift,
            mixed_alternating_tasks,
        ):
            trace = workload(seed=42)
            per_phase_distinct = []
            for start, end in (
                (0, 249),
                (250, 499),
                (500, 749),
                (750, 999),
                (1000, 1199),
            ):
                phase = [
                    request for request in trace
                    if start <= request.token_step <= end
                ]
                per_phase_distinct.append(len({
                    request.key for request in phase
                }))
            self.assertTrue(any(count > 24 for count in per_phase_distinct))


if __name__ == "__main__":
    unittest.main()
