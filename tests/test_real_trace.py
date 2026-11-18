import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from baseline import ExpertRequest, LRUCache, replay
from real_trace.collector import (
    GenerationMetadata,
    RouteTraceCollector,
    TRACE_FIELDS,
)
from real_trace.smoke_test import verify_local_files


class FakeMoE(torch.nn.Module):
    top_k = 2
    num_experts = 5

    def __init__(self):
        super().__init__()
        self.register_buffer("expert_bias", torch.zeros(5))
        self.router = torch.nn.Identity()

    def forward(self, router_logits):
        selected = torch.topk(
            torch.sigmoid(router_logits.float()) + self.expert_bias,
            self.top_k,
            dim=-1,
        ).indices
        return torch.zeros_like(router_logits), router_logits


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward = FakeMoE()


class FakeTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])

    def forward(self, input_ids):
        sequence_length = input_ids.shape[1]
        logits = torch.tensor(
            [[0.1, 3.0, -1.0, 0.5, 2.0]],
            dtype=torch.float32,
        ).expand(sequence_length, -1)
        for layer in self.layers:
            layer.feed_forward(logits)
        return logits


class FakeCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeTextModel()


class TraceCollectorTests(unittest.TestCase):
    def _collector(self):
        model = FakeCausalModel()
        collector = RouteTraceCollector(
            model=model,
            metadata=GenerationMetadata(
                request_id="unit-0001",
                prompt_category="unit_test",
                seed=42,
                max_new_tokens=2,
                do_sample=False,
                temperature=None,
                top_p=None,
            ),
            num_experts=5,
            expected_top_k=2,
        )
        collector.install()
        return model, collector

    def _capture_fake(self):
        model, collector = self._collector()
        with collector.capture_mode():
            output = model.model(input_ids=torch.zeros((1, 2), dtype=torch.long))
        collector.remove()
        collector.validate()
        return collector, output

    def test_captures_actual_top_k_selection_and_layer_mapping(self):
        collector, _ = self._capture_fake()
        self.assertEqual(len(collector.events), 2 * 2 * 2)
        self.assertEqual(
            {event["layer_index"] for event in collector.events},
            {0, 1},
        )
        self.assertEqual(
            {
                (event["token_index"], event["layer_index"])
                for event in collector.events
            },
            {(0, 0), (0, 1), (1, 0), (1, 1)},
        )
        self.assertEqual(
            [event["expert_id"] for event in collector.events[:2]],
            [1, 4],
        )
        for key in {(event["token_index"], event["layer_index"])
                    for event in collector.events}:
            group = [
                event for event in collector.events
                if (event["token_index"], event["layer_index"]) == key
            ]
            self.assertEqual(len(group), 2)
            self.assertEqual(len({event["expert_id"] for event in group}), 2)
            self.assertTrue(
                all(0 <= event["expert_id"] < 5 for event in group)
            )

    def test_csv_serialization_is_deterministic_and_has_fixed_columns(self):
        collector, _ = self._capture_fake()
        first = collector.csv_text("prefill")
        second = collector.csv_text("prefill")
        self.assertEqual(first, second)
        self.assertEqual(next(csv.reader(io.StringIO(first))), list(TRACE_FIELDS))
        self.assertEqual(len(list(csv.DictReader(io.StringIO(first)))), 8)
        self.assertEqual(collector.csv_text("decode").count("\n"), 1)
        self.assertEqual(len(list(csv.DictReader(
            io.StringIO(collector.csv_text())
        ))), 8)

    def test_cache_request_accounting_from_trace_events(self):
        collector, _ = self._capture_fake()
        requests = [
            ExpertRequest(
                token_step=event["token_index"],
                layer=event["layer_index"],
                expert=event["expert_id"],
                hot_phase=0,
            )
            for event in collector.events
        ]
        result = replay(requests, LRUCache(32))
        self.assertEqual(result["hits"] + result["misses"], len(requests))

    def test_instrumentation_does_not_change_forward_output(self):
        model, collector = self._collector()
        collector.remove()
        input_ids = torch.zeros((1, 2), dtype=torch.long)
        expected = model.model(input_ids=input_ids)
        collector.install()
        with collector.capture_mode():
            observed = model.model(input_ids=input_ids)
        collector.remove()
        collector.validate()
        self.assertTrue(torch.equal(expected, observed))


class IntegrityManifestTests(unittest.TestCase):
    @staticmethod
    def _blob_sha(data):
        header = b"blob " + str(len(data)).encode("ascii") + b"\0"
        return hashlib.sha1(header + data).hexdigest()

    def _create_fixture(self, directory):
        source_content = {
            "configuration_mobilemoe.py": b"config fixture\n",
            "modeling_mobilemoe.py": b"model fixture\n",
        }
        for name, content in source_content.items():
            (directory / name).write_bytes(content)
        weight_content = b"fixture weights"
        (directory / "model.safetensors").write_bytes(weight_content)
        manifest = {
            "schema_version": 1,
            "model_id": "fixture/model",
            "revision": "a" * 40,
            "expected_code_blobs": {
                name: self._blob_sha(content)
                for name, content in source_content.items()
            },
            "weights": {
                "filename": "model.safetensors",
                "bytes": len(weight_content),
                "sha256": hashlib.sha256(weight_content).hexdigest(),
            },
            "expected_architecture": {
                "num_experts": 3,
                "top_k": 2,
                "num_moe_layers": 2,
            },
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest, manifest_path

    def test_integrity_verification_accepts_pins_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest, manifest_path = self._create_fixture(directory)
            self.assertEqual(
                verify_local_files(directory, manifest_path),
                manifest,
            )

            source = directory / "modeling_mobilemoe.py"
            source.write_bytes(b"modified source\n")
            with self.assertRaisesRegex(RuntimeError, "Audited source changed"):
                verify_local_files(directory, manifest_path)

            source.write_bytes(b"model fixture\n")
            weight_path = directory / "model.safetensors"
            weight_path.write_bytes(b"short")
            with self.assertRaisesRegex(RuntimeError, "Weight size mismatch"):
                verify_local_files(directory, manifest_path)

            weight_path.write_bytes(b"changed weights")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                verify_local_files(directory, manifest_path)

    def test_integrity_verification_rejects_malformed_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _, manifest_path = self._create_fixture(directory)
            with self.assertRaises(FileNotFoundError):
                verify_local_files(directory, directory / "missing.json")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["revision"] = "not-a-pinned-revision"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "revision"):
                verify_local_files(directory, manifest_path)


if __name__ == "__main__":
    unittest.main()
