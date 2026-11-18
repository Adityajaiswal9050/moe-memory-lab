# MoE Memory Lab

MoE Memory Lab is a reproducible research prototype for studying expert reuse
and cache-policy behavior in Mixture-of-Experts (MoE) systems. It separates
three kinds of work:

1. **Synthetic routing experiments** with deterministic, explicitly defined
   request distributions.
2. **Real router observations** collected from an authorized, locally stored
   gated model.
3. **Cache replay simulations** that replay routing requests through software
   cache policies.

The synthetic workflows are the self-contained starting point. The proposed
expanded Days 1–7 code release also includes the original capture and analysis
code, but not model files or model-derived outputs. Synthetic experiments
require neither Hugging Face authentication nor model weights, a GPU, or
PyTorch. Real-trace observations and analyses remain on publication hold
pending review of the model's FAIR Noncommercial Research License.

This repository does not implement expert offloading in a model runtime.
Simulated misses, transfer bytes, and latency are not measurements of disk
traffic, inference latency, or tokens per second. No inference speedup is
claimed.

## Research motivation

Sparse MoE models route each token to a subset of experts in each MoE layer.
When expert weights cannot all be kept in fast memory, a runtime might need to
load experts as routing decisions are made. This lab studies routing reuse and
how simple cache policies respond to different request streams. Its replay
experiments count simulated cache hits and misses; they do not implement or
measure a production offloading system.

## Architecture and cache model

An MoE layer contains a router and multiple expert networks. For each token,
the router selects a top-k subset of experts. The synthetic baseline models
this as two unique expert requests per layer and token. Cache keys include
both the layer and its local expert ID, and the baseline uses one global cache
across layers.

The baseline compares `NO_CACHE`, LRU, and LFU. Later synthetic experiments
also evaluate exponentially decayed LFU and segmented LRU (SLRU). Each policy
replays the same ordered request sequence within an experiment.

### Synthetic baseline assumptions

- 6 layers and 32 experts per layer.
- 2 unique active experts per layer per token.
- 1,200 token steps with seed 42.
- 16 MiB modeled size per expert.
- Four hot experts per layer and phase; the hot set changes every 250 tokens.
- The original baseline samples uniformly from each phase's hot set and has no
  cold-tail requests. It is not a learned router or a realistic long-tail
  distribution.
- The baseline budgets are 128, 256, and 512 MiB.

The Day 3 workloads are separate synthetic constructions: static skew with a
cold tail, abrupt hot-set changes, gradual drift, and alternating task
patterns. They retain the per-layer top-k constraint and use fixed seeds.

### Metrics and interpretation

Hits, misses, miss rate, and cache occupancy are replay accounting. In the
synthetic baseline, modeled transferred MiB equals misses multiplied by the
assumed 16 MiB expert size. Its modeled I/O latency adds an assumed 100 ms per
miss; that constant is not measured disk latency and must not be interpreted
as inference latency.

For model-derived traces, the cache unit is estimated from expert tensor
metadata. A replayed miss multiplied by that size is still only modeled expert
traffic: the replay does not transfer model weights. Prefill requests are
serialized for replay even though real prefill routes a batch of tokens;
hardware may coalesce or issue expert loads concurrently. Decode requests are
also replayed, not timed as cache-backed inference.

## Experiments

| Day | Scope |
|---|---|
| 1 | Synthetic baseline with `NO_CACHE`, LRU, and LFU. |
| 2 | Synthetic phase and adaptation analysis, alternate request order, and repeated-seed/budget comparisons. |
| 3 | Audit of the original generator and comparison of four distinct synthetic workloads. |
| 4 | A smoke test that records actual router selections from a locally available gated model. |
| 5 | Routing-frequency, reuse, and cache-replay analysis of a real trace. |
| 6 | A limited multi-prompt routing pilot with reset-per-prompt and persistent replay scenarios. |
| 7 | Synthetic development sweep for decay settings, plus exploratory comparisons of decayed LFU and SLRU. |

Days 4–7 are summarized here only at the workflow level. No model-specific
quantitative findings are included in this README.

## Publication hold and model-license boundary

The following third-party assets and model-derived findings are **on hold**
until redistribution and publication permissions are reviewed under the FAIR
Noncommercial Research License:

- Actual model-routing traces and model-derived results from Days 4–7.
- Day 4–7 CSV/JSON reports and figures.
- Downloaded gated-model source, configuration, tokenizer assets, and weights.
- Quantitative claims based on those traces.

The local `.gitignore` excludes these assets from Git while review is pending;
it does not delete local files or grant permission to redistribute them. The
model weight file, credentials, and Hugging Face caches must not be committed.
The proposed expanded release contains original project instrumentation and
analysis code separately from those assets. The initial staged candidate does
not include `real_trace/model_integrity.json` or any gated model files. The
local manifest contains factual identifiers, revision, digests, byte count,
and architecture metadata—not model source or weights—and is ignored. Those
facts are not automatically copyrighted material, but applicable model terms
still require review. The approved MIT license covers original project code
only, not Meta's model or third-party assets.

## Public synthetic reproduction on Windows

Tested with Python 3.12 on Windows. The Day 1 baseline uses only the Python
standard library. The Day 2/3 analysis scripts and their public tests require
the pinned Matplotlib dependency; they do not require CUDA, PyTorch, gated
weights, or Hugging Face authentication.

In PowerShell, from the repository root of a clean checkout:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe baseline.py
```

To run the Day 2 and Day 3 analyses:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-synthetic.txt
.\.venv\Scripts\python.exe analysis.py
.\.venv\Scripts\python.exe day3_analysis.py
```

These scripts generate or replace their corresponding synthetic outputs under
`data/`, `results/`, and `figures/`. To avoid overwriting an existing
experiment, reproduce in a fresh checkout or copy first.

Run the public synthetic tests without gated assets:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_baseline.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_analysis.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_day3.py" -v
```

The proposed expanded release also has public standalone Day 4–7 unit tests.
They use synthetic traces, fake models, or temporary files; they need optional
Python dependencies but no gated model assets, Hugging Face authentication,
GPU hardware, or inference. Install the optional dependencies as described
below, then run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_day5.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_day6.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_day7.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_real_trace.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py" -v
```

The synthetic subset is 15 tests, the standalone Day 4–7 subset is 20 tests,
and default full discovery runs 35 tests. Any private-data Day 5 integration
checks are separate, excluded from normal discovery and the public allowlist,
and require the authorized local trace, config, and Safetensors weights.

## Optional gated-model workflow

This workflow is separate from public synthetic reproduction. The staged
candidate includes capture scripts and runtime requirements, but excludes
the integrity manifest, gated model files, and traces. Do not run inference
without authorized access, acceptance of the model terms, a trusted matching
integrity manifest placed locally at `real_trace/model_integrity.json`, and
any required experiment approval. Without that manifest, the scripts stop
before loading model code.

1. Review the model card and applicable license; confirm that using and
   retaining outputs is permitted for the intended noncommercial research.
2. Install the pinned CUDA PyTorch wheel from the official PyTorch CUDA 12.8
   wheel index, then install the remaining pinned runtime dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
   .\.venv\Scripts\python.exe -m pip install -r real_trace/requirements.txt
   ```

   The installed PyTorch version satisfies its matching pin. No system CUDA
   driver change is part of this workflow.
3. Obtain the exact approved model revision through the authorized channel and
   place its local files in the ignored `real_trace/model_source/` directory.
   Obtain the matching `real_trace/model_integrity.json` from a trusted audit
   record and place it locally; it is deliberately not included in this
   initial candidate. Never derive expected hashes from downloaded files. The
   manifest holds factual model/revision identifiers, audited custom-code
   Git-blob hashes, expected weight byte size and SHA-256, and expected
   routing architecture. It contains no third-party implementation or model
   weights. These facts are not automatically copyrighted material, but that
   is not legal clearance under the model terms.
4. Before loading, both `smoke_test.py` and `day6_pilot.py` fail closed if the
   manifest is missing or malformed, either audited custom source hash differs,
   or the weight size or SHA-256 differs. They also check the loaded
   expert/top-k/layer configuration. Only after these checks do the scripts
   use local-only loading and `trust_remote_code=True` for the reviewed code.
   Tokenizer loading remains local-only and does not enable remote code. The
   manifest's revision is audit provenance; these checks hash the two custom
   Python source files and the weights, but do not individually hash config or
   tokenizer files.
5. Check available RAM/VRAM before running the one-prompt smoke test. The
   smoke test uses evaluation mode, inference mode, batch size 1, and at most
   eight generated tokens. If CUDA OOM occurs during load or generation, it
   records the failure and stops. It does **not** automatically retry with
   CPU offloading or switch models.
6. Any additional inference pilot requires its own approval. Keep the
   resulting traces and model-derived analyses local until licensing review is
   complete.

The dependency pins are recorded in `real_trace/requirements.txt` in the full
local workspace. The synthetic-only dependency is recorded separately in
`requirements-synthetic.txt`.

Offline analysis is distinct from inference: Day 5 needs the authorized Day 4
trace, local model configuration, and Safetensors metadata; Day 6 and Day 7
need the Day 6 trace/report plus local configuration and expert-size inputs.
Those analyses do not run the model, but their inputs and derived outputs
remain local pending terms review. `smoke_test.py` and `day6_pilot.py` are the
scripts that perform model inference.

## Project structure

```text
baseline.py, analysis.py       Day 1–2 synthetic baseline and analysis
day3_workloads.py              Explicit synthetic workload generators
day3_analysis.py               Day 3 synthetic comparisons
day5_analysis.py               Day 5 trace analysis and cache replay
day6_analysis.py               Multi-prompt analysis and replay code
day7_analysis.py               Synthetic sweep and exploratory replay driver
day7_policies.py               Decayed LFU and SLRU policy implementations
tests/                         Unit tests
data/                          Synthetic request trace
results/                       Synthetic outputs; Day 4–7 outputs are ignored
figures/                       Synthetic figures; Day 4–7 figures are ignored
real_trace/                    Collector and capture code; local model assets,
                               traces, and integrity manifest are ignored and
                               not in the initial staged set
```

This describes the full local working tree. `PUBLIC_RELEASE_PLAN.md` preserves
the exact 31-file synthetic baseline allowlist and proposes 13 additions for a
44-file Days 1–7 candidate. The initial staged set excludes the integrity
manifest, leaving 43 existing candidate files plus `LICENSE`. Actual traces,
third-party model files, Day 4–7 results/figures, and model-derived claims
remain excluded pending applicable terms review.

## Limitations and future research

- Synthetic results depend on the constructed routing distributions and are not
  evidence of learned-model routing behavior.
- The original baseline has no cold tail; Day 3 adds synthetic cold-tail and
  drift scenarios but does not make them representative of real models.
- Real-trace experiments are small exploratory samples, not held-out
  evaluation. They must not be generalized to all prompts or MoE models.
- Prefill serialization is a replay approximation, not physical expert-load
  ordering.
- Modeled transfer bytes are not measured memory or disk traffic. No
  tokens-per-second or inference-latency improvement is established.
- A future study should use a pre-registered held-out prompt set and, if
  authorized, separately measure a real runtime's I/O and inference behavior.

## Project license

The approved MIT `LICENSE` applies only to original MoE Memory Lab
contributions. It does not apply to Meta's model or any third-party source,
weights, tokenizer/configuration assets, or model-derived outputs; their
separate terms continue to apply.
