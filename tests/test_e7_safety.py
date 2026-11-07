"""Small, CPU-only release gates for the E7 continuation machinery."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from posenerf.telemetry import atomic_torch_save
from posenerf.training import _checkpoint_sha256, _record_immutable_snapshot
from posenerf.utils import capture_rng_state, restore_rng_state


class E7SafetyTests(unittest.TestCase):
    def test_rng_round_trip_restores_the_next_cpu_draw(self):
        torch.manual_seed(91)
        state = capture_rng_state()
        expected = torch.rand(5)
        _ = torch.rand(7)
        restore_rng_state(state)
        self.assertTrue(torch.equal(expected, torch.rand(5)))

    def test_immutable_snapshot_manifest_is_hashed_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw:
            directory = Path(raw)
            snap = directory / "hierarchical_nerf_step_12001.pth"
            atomic_torch_save({"step": 12000, "tensor": torch.arange(3)}, snap)
            _record_immutable_snapshot(directory, snap)
            manifest = json.loads((directory / "immutable_snapshot_manifest.json").read_text())
            self.assertEqual(manifest["snapshots"][snap.name]["sha256"], _checkpoint_sha256(snap))
            with self.assertRaises(FileExistsError):
                _record_immutable_snapshot(directory, snap)

    def test_controlled_schedule_metadata_preserves_original_horizon(self):
        source = Path("posenerf/training.py").read_text(encoding="utf-8")
        self.assertIn('schedule_state.get("start_step"', source)
        self.assertIn('schedule_state.get("end_step"', source)
        self.assertIn('"extension_schedule"', source)

    def test_atomic_save_does_not_leave_a_temporary_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as raw:
            target = Path(raw) / "checkpoint.pth"
            atomic_torch_save({"finite": True}, target)
            self.assertTrue(target.is_file())
            self.assertFalse(target.with_suffix(".pth.tmp").exists())
