"""Run one guarded MobileMoE inference and capture its real router selections."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    from .collector import GenerationMetadata, RouteTraceCollector
else:
    from collector import GenerationMetadata, RouteTraceCollector


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "real_trace" / "model_source"
INTEGRITY_MANIFEST_PATH = ROOT / "real_trace" / "model_integrity.json"
TRACE_PATH = ROOT / "real_trace" / "first_real_trace.csv"
RESULTS_DIR = ROOT / "results" / "day4"
REPORT_PATH = RESULTS_DIR / "smoke_test_report.json"
REQUEST_ID = "day4-smoke-0001"
SEED = 42
MAX_NEW_TOKENS = 8
CATEGORY = "factual_qa"
PROMPT = "What is the boiling point of water at sea level?"
MIN_FREE_CUDA_BYTES = int(3.5 * 1024**3)
MIN_FREE_RAM_BYTES = 1 * 1024**3


def memory_snapshot():
    virtual = psutil.virtual_memory()
    result = {
        "system_ram_total_bytes": virtual.total,
        "system_ram_available_bytes": virtual.available,
    }
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        result.update({
            "cuda_name": torch.cuda.get_device_name(0),
            "cuda_total_bytes": total_bytes,
            "cuda_free_bytes": free_bytes,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
            "cuda_bf16_supported": torch.cuda.is_bf16_supported(),
        })
    return result


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def load_integrity_manifest(manifest_path: Path) -> dict:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Integrity manifest must contain a JSON object.")

    required_keys = {
        "schema_version",
        "model_id",
        "revision",
        "expected_code_blobs",
        "weights",
        "expected_architecture",
    }
    if set(manifest) != required_keys:
        raise ValueError("Integrity manifest has missing or unknown fields.")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Unsupported integrity manifest schema version.")
    if not isinstance(manifest["model_id"], str) or not manifest["model_id"]:
        raise ValueError("Integrity manifest model_id must be a non-empty string.")
    if (
        not isinstance(manifest["revision"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"])
    ):
        raise ValueError("Integrity manifest revision must be a 40-digit hex SHA.")

    code_blobs = manifest["expected_code_blobs"]
    expected_source_files = {
        "configuration_mobilemoe.py",
        "modeling_mobilemoe.py",
    }
    if not isinstance(code_blobs, dict) or set(code_blobs) != expected_source_files:
        raise ValueError("Integrity manifest has an unexpected source-file list.")
    for digest in code_blobs.values():
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{40}", digest)
        ):
            raise ValueError("Source-code blob digests must be 40-digit hex SHAs.")

    weights = manifest["weights"]
    if (
        not isinstance(weights, dict)
        or set(weights) != {"filename", "bytes", "sha256"}
    ):
        raise ValueError("Integrity manifest has invalid weight metadata.")
    if weights["filename"] != "model.safetensors":
        raise ValueError("Integrity manifest must pin model.safetensors.")
    if type(weights["bytes"]) is not int or weights["bytes"] <= 0:
        raise ValueError("Expected weight size must be a positive integer.")
    if not isinstance(weights["sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", weights["sha256"]
    ):
        raise ValueError("Weight SHA-256 must be a 64-digit hex digest.")

    architecture = manifest["expected_architecture"]
    architecture_keys = {"num_experts", "top_k", "num_moe_layers"}
    if (
        not isinstance(architecture, dict)
        or set(architecture) != architecture_keys
    ):
        raise ValueError("Integrity manifest has invalid architecture metadata.")
    if any(
        type(value) is not int or value <= 0
        for value in architecture.values()
    ):
        raise ValueError("Expected architecture values must be positive integers.")
    if architecture["top_k"] > architecture["num_experts"]:
        raise ValueError("Expected top_k cannot exceed num_experts.")
    return manifest


def verify_local_files(
    model_dir: Path = MODEL_DIR,
    manifest_path: Path = INTEGRITY_MANIFEST_PATH,
) -> dict:
    manifest = load_integrity_manifest(manifest_path)
    for filename, expected_blob in manifest["expected_code_blobs"].items():
        source_path = model_dir / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Audited source file not found: {filename}")
        actual_blob = git_blob_sha(source_path)
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"Audited source changed: {filename}; refusing to execute it."
            )

    weight_path = model_dir / manifest["weights"]["filename"]
    if not weight_path.is_file():
        raise FileNotFoundError(f"Expected local weights not found: {weight_path}")
    if weight_path.stat().st_size != manifest["weights"]["bytes"]:
        raise RuntimeError(
            f"Weight size mismatch: {weight_path.stat().st_size} bytes"
        )

    digest = hashlib.sha256()
    with weight_path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["weights"]["sha256"]:
        raise RuntimeError("Pinned model weight SHA-256 mismatch.")
    return manifest


def write_report(report):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def preflight_report(integrity: dict):
    memory = memory_snapshot()
    issues = []
    if not torch.cuda.is_available():
        issues.append("CUDA is unavailable; this smoke test requires the GPU.")
    elif memory["cuda_free_bytes"] < MIN_FREE_CUDA_BYTES:
        issues.append(
            f"CUDA free memory is below 3.5 GiB "
            f"({memory['cuda_free_bytes']} bytes)."
        )
    elif not memory["cuda_bf16_supported"]:
        issues.append("RTX device reports no BF16 support.")
    if memory["system_ram_available_bytes"] < MIN_FREE_RAM_BYTES:
        issues.append(
            "Available system RAM is below 1 GiB "
            f"({memory['system_ram_available_bytes']} bytes); refusing load."
        )

    return {
        "status": "blocked_preflight" if issues else "preflight_passed",
        "issues": issues,
        "model_id": integrity["model_id"],
        "revision": integrity["revision"],
        "weights_bytes": integrity["weights"]["bytes"],
        "weights_sha256": integrity["weights"]["sha256"],
        "memory_before_load": memory,
        "model_loaded": False,
        "inference_completed": False,
        "trace_validated": False,
        "prompt_content_saved": False,
    }


def main() -> int:
    integrity = verify_local_files()
    report = preflight_report(integrity)
    if report["issues"]:
        write_report(report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": 0},
            max_memory={0: "4GiB"},
            attn_implementation="eager",
        )
    except torch.cuda.OutOfMemoryError as error:
        report.update({
            "status": "cuda_oom_loading",
            "issues": [str(error)],
            "load_seconds": time.perf_counter() - load_start,
            "memory_at_failure": memory_snapshot(),
        })
        write_report(report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 3

    model_load_seconds = time.perf_counter() - load_start
    model.eval()
    expected_architecture = integrity["expected_architecture"]
    if (
        int(model.config.num_local_experts) != expected_architecture["num_experts"]
        or int(model.config.num_experts_per_tok) != expected_architecture["top_k"]
    ):
        raise RuntimeError(
            "Loaded model configuration does not match the reviewed manifest."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        trust_remote_code=False,
    )
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    prompt_length = int(encoded.shape[-1])
    input_device = model.get_input_embeddings().weight.device
    input_ids = encoded.to(input_device)
    attention_mask = torch.ones_like(input_ids)
    metadata = GenerationMetadata(
        request_id=REQUEST_ID,
        prompt_category=CATEGORY,
        seed=SEED,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )
    collector = RouteTraceCollector(
        model,
        metadata,
        num_experts=expected_architecture["num_experts"],
        expected_top_k=expected_architecture["top_k"],
    )
    collector.install()
    if (
        len(collector._layer_indices)
        != expected_architecture["num_moe_layers"]
    ):
        collector.remove()
        raise RuntimeError(
            f"Expected {expected_architecture['num_moe_layers']} MoE layers, "
            f"found "
            f"{len(collector._layer_indices)}."
        )

    inference_start = time.perf_counter()
    try:
        with torch.inference_mode(), collector.capture_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError as error:
        collector.remove()
        report.update({
            "status": "cuda_oom_inference",
            "issues": [str(error)],
            "model_loaded": True,
            "model_load_seconds": model_load_seconds,
            "memory_at_failure": memory_snapshot(),
        })
        write_report(report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 4
    finally:
        collector.remove()

    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    collector.validate()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    events_csv = collector.csv_text()
    TRACE_PATH.write_text(events_csv, encoding="utf-8", newline="")
    (RESULTS_DIR / "smoke_prefill.csv").write_text(
        collector.csv_text("prefill"), encoding="utf-8", newline=""
    )
    (RESULTS_DIR / "smoke_decode.csv").write_text(
        collector.csv_text("decode"), encoding="utf-8", newline=""
    )

    config = model.config
    report.update({
        "status": "complete",
        "issues": [],
        "model_loaded": True,
        "inference_completed": True,
        "trace_validated": True,
        "request_id": REQUEST_ID,
        "prompt_category": CATEGORY,
        "prompt_content_saved": False,
        "prompt_tokens": prompt_length,
        "generated_tokens": int(generated.shape[-1]) - prompt_length,
        "generation": {
            "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "batch_size": 1,
        },
        "model_config": {
            "model_type": config.model_type,
            "num_hidden_layers": config.num_hidden_layers,
            "moe_layer_indices": sorted(collector._layer_indices),
            "num_local_experts": config.num_local_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "torch_dtype": str(config.torch_dtype),
            "attention_implementation": "eager",
        },
        "routing_event_count": len(collector.events),
        "prefill_event_count": sum(
            event["phase"] == "prefill" for event in collector.events
        ),
        "decode_event_count": sum(
            event["phase"] == "decode" for event in collector.events
        ),
        "load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "memory_after_run": memory_snapshot(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "model_device_map": getattr(model, "hf_device_map", None),
        "trace_path": str(TRACE_PATH.relative_to(ROOT)),
    })
    write_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
