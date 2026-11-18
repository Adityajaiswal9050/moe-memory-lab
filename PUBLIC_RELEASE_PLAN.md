# Public Release Plan

## Scope and release gate

This plan keeps the verified 31-file **synthetic-only baseline** unchanged and
adds a separate proposal for a complete Days 1–7 release. The expanded
candidate is the union of that exact baseline and the 13 original-code and
integrity-metadata additions below: 44 files total. The first local staging
set excludes `real_trace/model_integrity.json`, leaving 43 existing candidate
files plus the new `LICENSE`. Staging is not authorization to commit or
publish.

The expanded proposal contains project-authored instrumentation and analysis,
not the gated model implementation, weights, actual routing traces, or
model-derived results and figures. The listed Day 4–7 code is generally
dependent on local gated-model files or private traces to run. Synthetic-only
setup and tests remain usable without those assets.

The project owner approved **MIT** for original MoE Memory Lab code. The new
`LICENSE` applies only to original project contributions; it does not relicense
Meta's model or any third-party materials. This plan does not authorize a
commit or publication.

## Verified 31-file synthetic baseline allowlist

Keep this exact 31-file list as the safe baseline candidate:

```text
.gitignore
README.md
PUBLIC_RELEASE_PLAN.md
requirements-synthetic.txt
baseline.py
analysis.py
day3_workloads.py
day3_analysis.py
tests/test_baseline.py
tests/test_analysis.py
tests/test_day3.py
data/synthetic_trace.csv
results/adaptation_50_tokens.csv
results/alternative_order.csv
results/baseline.csv
results/phase_analysis.csv
results/seed_summary.csv
results/day3/baseline_audit_histogram.csv
results/day3/baseline_audit_phase.csv
results/day3/baseline_audit_summary.csv
results/day3/repeated_seed_variance.csv
results/day3/workload_frequency_histogram.csv
results/day3/workload_phase_cache.csv
results/day3/workload_phase_fractions.csv
results/day3/workload_seed_summary.csv
results/day3/workload_summary.csv
figures/phase_hit_rate_and_transfer.png
figures/day3/abrupt_changes_hit_and_transfer.png
figures/day3/gradual_drift_hit_and_transfer.png
figures/day3/mixed_alternating_tasks_hit_and_transfer.png
figures/day3/static_skewed_hit_and_transfer.png
```

These are content-reviewed baseline candidates. The owner approved MIT for
original project code; the explicit first staging set is the baseline plus
the 12 non-manifest additions and `LICENSE`.

## Keep private pending license review

Do not include or redistribute the following model-derived assets or findings
until the FAIR Noncommercial Research License and the intended publication
context have been reviewed:

```text
real_trace/first_real_trace.csv
real_trace/model_source/**
real_trace/model_code/**
results/day4/**
results/day5/**
results/day6/**
results/day7/**
figures/day4/**
figures/day5/**
figures/day6/**
figures/day7/**
```

This hold covers actual router traces, derived tables/reports/figures, and
downloaded gated-model configuration, tokenizer, source, and weight files.
The directories remain local; `.gitignore` excludes them without deleting
them. Public-facing materials should not report quantitative conclusions
from these experiments before review.

## Expanded additions to the 31-file baseline

These 12 project-authored source, test, and dependency files plus the
integrity manifest are proposed additions. They were reviewed individually
for apparent project authorship, credentials, personal paths, and copied
gated implementation. No credentials, personal paths, or substantial
matching nontrivial source lines were detected. This is a source review, not
a legal opinion or a substitute for confirming authorship and permissions.

```text
day5_analysis.py
day6_analysis.py
day7_analysis.py
day7_policies.py
real_trace/collector.py
real_trace/smoke_test.py
real_trace/day6_pilot.py
real_trace/model_integrity.json
real_trace/requirements.txt
tests/test_day5.py
tests/test_day6.py
tests/test_day7.py
tests/test_real_trace.py
```

| File | Assessment and runtime inputs |
|---|---|
| `real_trace/collector.py` | Original routing instrumentation. Imports PyTorch; requires a compatible loaded model to collect anything, but contains no model implementation or weights. |
| `real_trace/smoke_test.py` | Original guarded one-prompt runner. Reads integrity expectations from `real_trace/model_integrity.json`, verifies audited source Git-blob hashes and exact weight size/SHA-256 before loading, then checks the loaded architecture. It requires local model files and runtime dependencies; it uses `trust_remote_code=True` only after verification and loads only local files. |
| `real_trace/day6_pilot.py` | Original multi-prompt runner with generic prompt categories. Requires the same local model and runtime; it performs inference and must only be run with separate authorization. |
| `real_trace/model_integrity.json` | Factual model identity/revision, source-file digests, weight checksum/byte count, and architecture expectations. It contains no model source or weights. The runner verifies the two listed custom-code files and weights; the revision is audit provenance, and config/tokenizer files are not individually hashed. These facts are not automatically copyrighted material; applicable model terms still need review. |
| `day5_analysis.py` | Original analysis/replay implementation. Requires the Day 4 trace, local model config, and Safetensors metadata to calculate expert sizes. |
| `day6_analysis.py` | Original analysis/replay implementation. Requires the Day 6 trace/report and local model config and Safetensors metadata. |
| `day7_analysis.py` | Original synthetic tuning and exploratory replay driver. Requires Day 6 trace/report and local Safetensors metadata for its real-trace evaluation. |
| `day7_policies.py` | Original standalone policy implementations; standard-library only and does not require gated assets. |
| `tests/test_day5.py` | Original unit tests using a small synthetic trace and synthetic tensor metadata; they need no private files and make no claims about real measurements. |
| `tests/test_day6.py` | Original in-memory fixtures; no model weights or actual routing trace are required. Analysis dependencies are needed for imports. |
| `tests/test_day7.py` | Original policy/replay fixtures; no actual routing trace or weights are needed for the test cases. Analysis dependencies are needed for imports. |
| `tests/test_real_trace.py` | Original fake-model and temporary integrity-manifest tests; require PyTorch but not gated files, CUDA, or real-model inference. |
| `real_trace/requirements.txt` | Original dependency pins for the optional model workflow; includes no model assets or credentials. |

`tests/day5_private_validation.py` preserves four original assertions against
the private Day 4 trace/configuration/weight metadata. It is intentionally
ignored by Git and is not part of either public allowlist. It can be run only
in the authorized local workspace. The public Day 5 tests use independent
synthetic fixtures instead.

The proposed all-days candidate includes the factual integrity manifest as a
separate metadata file, but the first local staging set deliberately excludes
`real_trace/model_integrity.json`. It is locally ignored for now. Without a
trusted matching manifest at that path, the capture runners fail closed
before loading custom model code.

The manifest's revision, SHA digests, byte count, and architecture fields are
factual integrity metadata, not copies of third-party source or weights. They
are kept separate from the runner so its code can be published and reviewed
independently. This distinction is not legal clearance: review applicable
model terms before publishing the metadata or any model-derived experiment
outputs.

The dependency distinction is:

- The 15 Day 1–3 synthetic tests need the documented synthetic environment.
- The public Day 4–7 tests use fake models, synthetic traces, or temporary
  integrity files. They need optional Python dependencies, but no gated model
  files, Hugging Face authentication, or GPU hardware.
- `test_real_trace.py` uses a fake CPU model and needs PyTorch, but does not
  access model files or perform real-model inference.
- Day 6/7 unit tests use in-memory fixtures, but importing their analysis
  modules requires the optional analysis dependencies.
- The optional `day5_private_validation.py` tests require the local Day 4
  trace, model config, and Safetensors weights. Day 5 analysis and the Day 6/7
  analysis entry points likewise require their documented private inputs.
- `smoke_test.py` and `day6_pilot.py` perform real inference. Publication of
  their code does not grant model access or authorize running the workflow.

`real_trace/model_source/` and `real_trace/model_code/` contain third-party
gated model materials and are not original project code. They must not be
included in either allowlist or relicensed under MIT. The same hold applies
to actual traces and derived outputs listed above.

## Local-only and generated files

Never include local credentials, caches, virtual environments, or bytecode:

```text
.venv/**
__pycache__/**
.env
.env.* (except an intentionally sanitized root .env.example)
.hf_token
.huggingface/**
.cache/huggingface/**
.netrc
credentials.json
real_trace/**/*.safetensors
real_trace/**/*.bin
real_trace/**/*.pt
real_trace/**/*.pth
real_trace/model_integrity.json
tests/day5_private_validation.py
```

These are excluded by `.gitignore`. The home-directory Hugging Face token is
outside this repository and must remain outside it.

## Reproduction commands for a new Windows user

Run these commands from the repository root in PowerShell. The baseline uses
only the Python standard library. Day 2/3 analysis and plots use the pinned
Matplotlib dependency. No gated-model access, weights, CUDA, or PyTorch are
required.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe baseline.py
.\.venv\Scripts\python.exe -m pip install -r requirements-synthetic.txt
.\.venv\Scripts\python.exe analysis.py
.\.venv\Scripts\python.exe day3_analysis.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_baseline.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_analysis.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_day3.py" -v
```

The synthetic subset contains 15 tests. The public Day 4–7 tests contain 20
tests and need optional Python dependencies, but no model files or GPU. The
default full suite contains 35 tests. The four private Day 5 validation tests
are separate and excluded from normal discovery and the public allowlist. The
synthetic reproduction was run in an isolated temporary copy
containing only the Day 1–3 scripts and tests. All 15 tests passed, and the
generated synthetic CSVs and figures matched the current outputs byte for
byte with Python 3.12 and Matplotlib 3.11.2 on Windows. No nondeterministic
differences were observed in that environment; byte-identical plots across
other operating systems or library versions are not guaranteed. No model files
or excluded assets were needed.

## License scope

The project owner approved MIT for original MoE Memory Lab code, and the
staged `LICENSE` applies only to those original contributions. It does not
apply to Meta's model or its source, weights, tokenizer/configuration assets,
or model-derived outputs. Any model-specific metadata remains subject to
applicable model-term review; no legal clearance for model-derived results is
claimed.
