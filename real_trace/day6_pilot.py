"""Collect a four-prompt MobileMoE routing pilot from the pinned local model."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    from .collector import GenerationMetadata, RouteTraceCollector, TRACE_FIELDS
    from .smoke_test import (
        MAX_NEW_TOKENS,
        MIN_FREE_CUDA_BYTES,
        MIN_FREE_RAM_BYTES,
        MODEL_DIR,
        memory_snapshot,
        verify_local_files,
    )
else:
    from collector import GenerationMetadata, RouteTraceCollector, TRACE_FIELDS
    from smoke_test import (
        MAX_NEW_TOKENS,
        MIN_FREE_CUDA_BYTES,
        MIN_FREE_RAM_BYTES,
        MODEL_DIR,
        memory_snapshot,
        verify_local_files,
    )


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "day6"
TRACE_PATH = RESULTS_DIR / "routing_events.csv"
REPORT_PATH = RESULTS_DIR / "pilot_report.json"
SEED = 42
TEMPERATURE = 1.0
TOP_P = 1.0
PROMPTS = (
    (
        "coding",
        "day6-coding-0001",
        "Write a Python function that returns the factorial of a "
        "nonnegative integer.",
    ),
    (
        "mathematics",
        "day6-mathematics-0001",
        "What is 17 times 23? Show the result.",
    ),
    (
        "general_knowledge",
        "day6-general-knowledge-0001",
        "What is the capital city of Japan?",
    ),
    (
        "scientific_reasoning",
        "day6-scientific-reasoning-0001",
        "Why does ice float on liquid water?",
    ),
)


def write_report(report: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_trace(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(TRACE_FIELDS)
    with TRACE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run_preflight(integrity: dict) -> dict:
    memory = memory_snapshot()
    issues = []
    if not torch.cuda.is_available():
        issues.append("CUDA is unavailable; this pilot requires the GPU.")
    elif memory["cuda_free_bytes"] < MIN_FREE_CUDA_BYTES:
        issues.append(
            "CUDA free memory is below 3.5 GiB "
            f"({memory['cuda_free_bytes']} bytes)."
        )
    elif not memory["cuda_bf16_supported"]:
        issues.append("The GPU reports no BF16 support.")
    if memory["system_ram_available_bytes"] < MIN_FREE_RAM_BYTES:
        issues.append(
            "Available system RAM is below 1 GiB "
            f"({memory['system_ram_available_bytes']} bytes)."
        )
    return {
        "status": "blocked_preflight" if issues else "preflight_passed",
        "issues": issues,
        "model_id": integrity["model_id"],
        "revision": integrity["revision"],
        "weights_bytes": integrity["weights"]["bytes"],
        "memory_before_load": memory,
        "model_loaded": False,
        "inference_completed": False,
        "prompt_content_saved": False,
    }


def main() -> int:
    integrity = verify_local_files()
    report = run_preflight(integrity)
    architecture = integrity["expected_architecture"]
    report.update({
        "request_order": [
            {"prompt_category": category, "request_id": request_id}
            for category, request_id, _ in PROMPTS
        ],
        "generation": {
            "seed": SEED,
            "batch_size": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "use_cache": True,
        },
        "routing": {
            "num_moe_layers": architecture["num_moe_layers"],
            "num_experts": architecture["num_experts"],
            "top_k": architecture["top_k"],
            "prefill_and_decode_separate": True,
        },
    })
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
            "memory_at_failure": memory_snapshot(),
        })
        write_report(report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 3

    model_load_seconds = time.perf_counter() - load_start
    model.eval()
    if (
        int(model.config.num_local_experts) != architecture["num_experts"]
        or int(model.config.num_experts_per_tok) != architecture["top_k"]
    ):
        raise RuntimeError(
            "Loaded model configuration does not match the reviewed manifest."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        trust_remote_code=False,
    )
    events = []
    request_reports = []

    for category, request_id, prompt in PROMPTS:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        prompt_length = int(encoded.shape[-1])
        input_ids = encoded.to(model.get_input_embeddings().weight.device)
        attention_mask = torch.ones_like(input_ids)
        metadata = GenerationMetadata(
            request_id=request_id,
            prompt_category=category,
            seed=SEED,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        collector = RouteTraceCollector(
            model,
            metadata,
            num_experts=architecture["num_experts"],
            expected_top_k=architecture["top_k"],
        )
        collector.install()
        if len(collector._layer_indices) != architecture["num_moe_layers"]:
            collector.remove()
            raise RuntimeError(
                f"Expected {architecture['num_moe_layers']} MoE layers, found "
                f"{len(collector._layer_indices)}."
            )

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        inference_start = time.perf_counter()
        try:
            with torch.inference_mode(), collector.capture_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
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
                "completed_requests": request_reports,
                "routing_event_count": len(events),
                "memory_at_failure": memory_snapshot(),
                "partial_trace_path": (
                    str(TRACE_PATH.relative_to(ROOT)) if events else None
                ),
            })
            if events:
                write_trace(events)
            write_report(report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 4
        finally:
            collector.remove()

        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start
        collector.validate()
        request_events = [dict(event) for event in collector.events]
        for event in request_events:
            event["request_order"] = len(events)
            events.append(event)
        request_reports.append({
            "request_id": request_id,
            "prompt_category": category,
            "prompt_tokens": prompt_length,
            "generated_tokens": int(generated.shape[-1]) - prompt_length,
            "routing_event_count": len(request_events),
            "prefill_event_count": sum(
                event["phase"] == "prefill" for event in request_events
            ),
            "decode_event_count": sum(
                event["phase"] == "decode" for event in request_events
            ),
            "inference_seconds": inference_seconds,
            "trace_validated": True,
            "prompt_content_saved": False,
            "completion_content_saved": False,
        })
        write_trace(events)
        report.update({
            "status": "in_progress",
            "model_loaded": True,
            "model_load_seconds": model_load_seconds,
            "completed_requests": request_reports,
            "routing_event_count": len(events),
            "trace_path": str(TRACE_PATH.relative_to(ROOT)),
        })
        write_report(report)

    torch.cuda.synchronize()
    config = model.config
    report.update({
        "status": "complete",
        "issues": [],
        "model_loaded": True,
        "inference_completed": True,
        "trace_validated": True,
        "completed_requests": request_reports,
        "routing_event_count": len(events),
        "prefill_event_count": sum(
            event["phase"] == "prefill" for event in events
        ),
        "decode_event_count": sum(
            event["phase"] == "decode" for event in events
        ),
        "model_load_seconds": model_load_seconds,
        "memory_after_run": memory_snapshot(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "model_device_map": getattr(model, "hf_device_map", None),
        "model_config": {
            "model_type": config.model_type,
            "num_hidden_layers": config.num_hidden_layers,
            "num_moe_layers": NUM_MOE_LAYERS,
            "num_local_experts": config.num_local_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "torch_dtype": str(config.torch_dtype),
            "attention_implementation": "eager",
        },
        "trace_path": str(TRACE_PATH.relative_to(ROOT)),
    })
    write_trace(events)
    write_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
