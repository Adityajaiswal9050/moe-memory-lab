"""Hooks for capturing MobileMoE router selections from actual model forwards."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Any

import torch
from torch.overrides import TorchFunctionMode


TRACE_FIELDS = (
    "request_order",
    "request_id",
    "forward_index",
    "phase",
    "phase_token_index",
    "token_index",
    "layer_index",
    "expert_id",
    "selected_rank",
    "prompt_category",
    "seed",
    "max_new_tokens",
    "do_sample",
    "temperature",
    "top_p",
)


@dataclass(frozen=True)
class GenerationMetadata:
    request_id: str
    prompt_category: str
    seed: int
    max_new_tokens: int
    do_sample: bool
    temperature: float | None
    top_p: float | None


class RouteTraceCollector:
    """Records the actual top-k indices returned during each model forward.

    MobileMoETextMoe returns its router logits but not the selected IDs.
    Its reviewed implementation selects experts with torch.topk. A
    TorchFunctionMode observes that call and records the returned indices
    unchanged. Module pre/post hooks provide the corresponding layer mapping.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        metadata: GenerationMetadata,
        num_experts: int,
        expected_top_k: int,
    ) -> None:
        self.model = model
        self.metadata = metadata
        self.num_experts = num_experts
        self.expected_top_k = expected_top_k
        self.events: list[dict[str, Any]] = []
        self._handles = []
        self._layer_indices: set[int] = set()
        self._forward_index = -1
        self._current_phase = ""
        self._current_phase_token_indices: list[int] = []
        self._current_token_indices: list[int] = []
        self._prompt_length: int | None = None
        self._decode_calls = 0
        self._active_layer_index: int | None = None
        self._captured_topk: torch.Tensor | None = None

    def capture_mode(self):
        return _TopKCaptureMode(self)

    def install(self) -> None:
        text_model = getattr(self.model, "model", None)
        if text_model is None or not hasattr(text_model, "layers"):
            raise ValueError("Expected MobileMoEForCausalLM with model.layers")

        self._handles.append(
            text_model.register_forward_pre_hook(
                self._on_model_forward, with_kwargs=True
            )
        )
        moe_count = 0
        for layer_index, layer in enumerate(text_model.layers):
            feed_forward = getattr(layer, "feed_forward", None)
            if feed_forward is None or not hasattr(feed_forward, "router"):
                continue
            if getattr(feed_forward, "top_k", None) != self.expected_top_k:
                raise ValueError(
                    f"Unexpected top-k on layer {layer_index}: "
                    f"{getattr(feed_forward, 'top_k', None)}"
                )
            if getattr(feed_forward, "num_experts", None) != self.num_experts:
                raise ValueError(
                    f"Unexpected expert count on layer {layer_index}: "
                    f"{getattr(feed_forward, 'num_experts', None)}"
                )
            self._handles.append(
                feed_forward.register_forward_pre_hook(
                    self._make_moe_pre_hook(layer_index)
                )
            )
            self._handles.append(
                feed_forward.register_forward_hook(self._make_moe_hook(layer_index))
            )
            self._layer_indices.add(layer_index)
            moe_count += 1
        if not moe_count:
            self.remove()
            raise ValueError("No MoE feed-forward layers found")

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _on_model_forward(self, module, args, kwargs):
        del module
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("Collector requires 2D input_ids for batch size one")
        if input_ids.shape[0] != 1:
            raise ValueError("Collector currently supports batch size one only")

        sequence_length = int(input_ids.shape[1])
        self._forward_index += 1
        if self._forward_index == 0:
            self._current_phase = "prefill"
            self._prompt_length = sequence_length
            self._current_phase_token_indices = list(range(sequence_length))
            self._current_token_indices = list(range(sequence_length))
        else:
            self._current_phase = "decode"
            self._current_phase_token_indices = [self._decode_calls]
            self._current_token_indices = [
                (self._prompt_length or 0) + self._decode_calls
            ]
            self._decode_calls += 1
            if sequence_length != 1:
                raise ValueError(
                    "Decode is expected to route exactly one input token"
                )

    def _make_moe_pre_hook(self, layer_index: int):
        def begin_capture(module, inputs):
            del module, inputs
            if self._active_layer_index is not None:
                raise ValueError("Nested MoE forward calls are not supported")
            self._active_layer_index = layer_index
            self._captured_topk = None

        return begin_capture

    def _make_moe_hook(self, layer_index: int):
        def capture(module, inputs, output):
            del inputs
            try:
                if self._active_layer_index != layer_index:
                    raise ValueError(
                        f"Top-k capture layer mismatch: active "
                        f"{self._active_layer_index}, hook {layer_index}"
                    )
                if not isinstance(output, tuple) or len(output) != 2:
                    raise ValueError(
                        f"MoE layer {layer_index} did not return "
                        "(output, router_logits)"
                    )
                router_logits = output[1]
                if not isinstance(router_logits, torch.Tensor):
                    raise ValueError(
                        f"Layer {layer_index} router logits are not a tensor"
                    )
                if (
                    router_logits.ndim != 2
                    or router_logits.shape[1] != self.num_experts
                ):
                    raise ValueError(
                        f"Unexpected router-logit shape at layer {layer_index}: "
                        f"{tuple(router_logits.shape)}"
                    )
                if router_logits.shape[0] != len(self._current_token_indices):
                    raise ValueError(
                        f"Token alignment mismatch at layer {layer_index}: "
                        f"{router_logits.shape[0]} rows versus "
                        f"{len(self._current_token_indices)} inputs"
                    )
                if self._captured_topk is None:
                    raise ValueError(
                        f"No actual torch.topk result captured at layer "
                        f"{layer_index}"
                    )

                selected = self._captured_topk
                if tuple(selected.shape) != (
                    len(self._current_token_indices),
                    self.expected_top_k,
                ):
                    raise ValueError(
                        f"Unexpected selected-index shape at layer {layer_index}: "
                        f"{tuple(selected.shape)}"
                    )
                selected_cpu = selected.to(device="cpu").tolist()
                for row_index, expert_ids in enumerate(selected_cpu):
                    if len(set(expert_ids)) != self.expected_top_k:
                        raise ValueError(
                            "Duplicate expert IDs in a token's top-k"
                        )
                    if any(
                        expert_id < 0 or expert_id >= self.num_experts
                        for expert_id in expert_ids
                    ):
                        raise ValueError(
                            "Router selected an out-of-range expert ID"
                        )

                    for rank, expert_id in enumerate(expert_ids):
                        self.events.append({
                            "request_order": len(self.events),
                            "request_id": self.metadata.request_id,
                            "forward_index": self._forward_index,
                            "phase": self._current_phase,
                            "phase_token_index": (
                                self._current_phase_token_indices[row_index]
                            ),
                            "token_index": self._current_token_indices[row_index],
                            "layer_index": layer_index,
                            "expert_id": expert_id,
                            "selected_rank": rank,
                            "prompt_category": self.metadata.prompt_category,
                            "seed": self.metadata.seed,
                            "max_new_tokens": self.metadata.max_new_tokens,
                            "do_sample": self.metadata.do_sample,
                            "temperature": self.metadata.temperature,
                            "top_p": self.metadata.top_p,
                        })
            finally:
                self._active_layer_index = None
                self._captured_topk = None

        return capture

    def validate(self) -> None:
        if not self.events:
            raise ValueError("No router events captured")

        by_token_layer: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
        for event in self.events:
            key = (
                event["phase"],
                event["forward_index"],
                event["token_index"],
                event["layer_index"],
            )
            by_token_layer.setdefault(key, []).append(event)

        for key, events in by_token_layer.items():
            expert_ids = [event["expert_id"] for event in events]
            ranks = sorted(event["selected_rank"] for event in events)
            if len(events) != self.expected_top_k:
                raise ValueError(
                    f"{key} has {len(events)} router selections, expected "
                    f"{self.expected_top_k}"
                )
            if len(set(expert_ids)) != self.expected_top_k:
                raise ValueError(f"{key} contains repeated expert IDs")
            if ranks != list(range(self.expected_top_k)):
                raise ValueError(f"{key} has invalid selected ranks")

        token_layers: dict[tuple[str, int, int], set[int]] = {}
        for phase, forward_index, token_index, layer_index in by_token_layer:
            token_key = (phase, forward_index, token_index)
            token_layers.setdefault(token_key, set()).add(layer_index)
        for token_key, layer_indices in token_layers.items():
            if layer_indices != self._layer_indices:
                raise ValueError(
                    f"{token_key} has route events for layers "
                    f"{sorted(layer_indices)}, expected "
                    f"{sorted(self._layer_indices)}"
                )

        if [event["request_order"] for event in self.events] != list(
            range(len(self.events))
        ):
            raise ValueError("Router event ordering is not contiguous")

    def csv_text(self, phase: str | None = None) -> str:
        if phase is not None and phase not in ("prefill", "decode"):
            raise ValueError("phase must be prefill, decode, or None")
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=TRACE_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            event for event in self.events
            if phase is None or event["phase"] == phase
        )
        return output.getvalue()


class _TopKCaptureMode(TorchFunctionMode):
    def __init__(self, collector: RouteTraceCollector) -> None:
        super().__init__()
        self.collector = collector

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        result = func(*args, **kwargs)
        if func is torch.topk and self.collector._active_layer_index is not None:
            if self.collector._captured_topk is not None:
                raise ValueError(
                    "More than one torch.topk call occurred inside an MoE layer"
                )
            requested_k = kwargs.get("k")
            if requested_k is None and len(args) > 1:
                requested_k = args[1]
            if requested_k != self.collector.expected_top_k:
                raise ValueError(
                    f"Observed top-k={requested_k}, expected "
                    f"{self.collector.expected_top_k}"
                )
            self.collector._captured_topk = result.indices.detach()
        return result
