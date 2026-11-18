# Synthetic benchmark results

> **Evidence boundary:** All figures and numeric results here are generated from **synthetic request streams**, not real model-routing traces. “Modeled transfer” means cache misses multiplied by an assumed expert size; it is **not actual disk/GPU traffic** or a measured tokens-per-second improvement.

[← Back to overview](../README.md)

## Experimental design

| Property | Day 1–2 baseline | Day 3 workload sweep |
|:--|:--|:--|
| MoE layers | 6 | 6 |
| Available experts per layer | 32 | 32 |
| Experts selected per token/layer | 2 | 2 |
| Requests per trace | 14,400 | 14,400 |
| Assumed expert size | 16 MiB | 16 MiB |
| Cache policy comparison | No cache, LRU, LFU | No cache, LRU, LFU |
| Seeds | Seed 42 baseline; multiple seeds in Day 2 | 42, 43, 44, 45, 46 |
| Main issue | Generator has no cold tail | Explicit workload shifts and cold-tail cases |

The replay preserves layer-specific expert identity in cache keys. At any tested budget, request counts are the same for every policy.

## 1. Why the first baseline was misleading

Day 1's generator sampled uniformly from a small, per-phase “hot” pool without any cold-tail requests. With a 512 MiB cache, the following result appeared:

| Day 1 budget | Policy | Hit rate | Misses / 14,400 |
|--:|:--|--:|--:|
| 128 MiB | LRU | 0.00% | 14,400 |
| 256 MiB | LRU | 52.17% | 6,887 |
| 256 MiB | LFU | 18.49% | 11,737 |
| 512 MiB | **LRU** | **99.17%** | **120** |
| 512 MiB | LFU | 31.17% | 9,912 |

The near-perfect LRU result is a **workload-generator artifact**, not evidence of a real-world inference improvement. The Day 3 audit confirmed a **0% cold-tail request fraction** in the baseline. Across seeds, the generated baseline visited different experts over phases, but did not sample a realistic long tail within phases.

**Source:** [`results/baseline.csv`](../results/baseline.csv) · [baseline audit](../results/day3/baseline_audit_summary.csv)

## 2. Four different synthetic routing regimes

Day 3 adds workloads intended to stress cache policies differently:

1. **Static skew:** stable popular experts with approximately 10% cold-tail traffic.
2. **Abrupt changes:** the popular working set jumps between phases.
3. **Gradual drift:** expert popularity changes progressively.
4. **Alternating task patterns:** repeated switching among distinct synthetic request patterns.

These are constructed scenarios—not inferred real-MoE task categories. The generator parameters and seeds are defined in [`day3_workloads.py`](../day3_workloads.py).

### Comparison at 512 MiB

Mean cache hit rates across **five seeds (42–46)**:

| Workload | LRU | LFU | Difference, LFU − LRU |
|:--|--:|--:|--:|
| Static skew | 87.84% | **90.26%** | +2.42 pp |
| Abrupt changes | **87.20%** | 28.44% | −58.76 pp |
| Gradual drift | **42.83%** | 14.11% | −28.72 pp |
| Alternating tasks | **42.60%** | 33.83% | −8.78 pp |

At 256 MiB, the corresponding LRU/LFU mean hit rates were:

| Workload | LRU | LFU |
|:--|--:|--:|
| Static skew | 40.93% | **55.44%** |
| Abrupt changes | **40.77%** | 15.94% |
| Gradual drift | **18.16%** | 5.07% |
| Alternating tasks | **18.13%** | 12.11% |

These comparisons illustrate the trade-off: LFU's accumulated frequency can help under stable skew, while LRU adapts more readily when the working set changes. The difference depends on workload and budget.

<details>
<summary><strong>Direct data and figures</strong></summary>

- [Day 3 five-seed statistics (CSV)](../results/day3/repeated_seed_variance.csv)
- [Day 3 workload composition (CSV)](../results/day3/workload_summary.csv)
- [Static skew plot](../figures/day3/static_skewed_hit_and_transfer.png)
- [Abrupt changes plot](../figures/day3/abrupt_changes_hit_and_transfer.png)
- [Gradual drift plot](../figures/day3/gradual_drift_hit_and_transfer.png)
- [Alternating task plots](../figures/day3/mixed_alternating_tasks_hit_and_transfer.png)
</details>

## 3. What the metrics do and do not show

For an assumed expert size `S = 16 MiB` and a given cache policy:

```text
cache hit rate       = hits / requests
cache miss rate      = misses / requests
modeled transferred  = misses × 16 MiB
modeled I/O latency  = misses × 100 ms  [early baseline assumption only]
```

The `100 ms` value is hypothetical. It is not disk throughput, PCIe latency, GPU copy time, or end-to-end inference latency. No model weights are loaded or transferred during these synthetic cache experiments.

**Comparability caution:** Day 1's original baseline and Day 3's workloads are different request distributions. Comparing their absolute cache metrics does not isolate the effect of a replacement policy.

## 4. What came after the public synthetic experiments?

Days 4–7 add **original Python implementations** for collecting authorized real-router events and replaying them against LRU, LFU, decayed LFU, and segmented LRU (SLRU). The gated-model traces and derived metrics are deliberately not included in this public release, pending the applicable model-license review.

In particular, decayed LFU and SLRU are established policy families. **No algorithmic novelty or general superiority is claimed here.**

## 5. Reproduce

On Windows with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-synthetic.txt
.\.venv\Scripts\python.exe baseline.py
.\.venv\Scripts\python.exe analysis.py
.\.venv\Scripts\python.exe day3_analysis.py
```

Run these commands in a **fresh checkout or a copy**, since experiment drivers write CSV and image outputs. Use the associated scripts and fixed seeds for full reproducibility.

[Read the architecture and methodology →](METHODOLOGY.md)
