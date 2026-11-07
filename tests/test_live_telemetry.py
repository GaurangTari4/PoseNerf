import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from posenerf.scene import camera_frustum_vertices, predicted_scene_snapshot, scene_snapshot
from posenerf.telemetry import RunTelemetry, atomic_torch_save, atomic_write_json, read_latest
from posenerf.dashboard import PAGE, _payload, _runs


class LiveTelemetryTests(unittest.TestCase):
    def test_atomic_status_schema_and_reconnect(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            self.assertIsNone(read_latest(root))
            telemetry = RunTelemetry(root, run_id="fixture")
            telemetry.status("running", 3, total_steps=10)
            latest = read_latest(root)
            self.assertEqual(latest["state"], "running")
            self.assertEqual(latest["step"], 3)
            self.assertEqual(latest["run_id"], "fixture")
            self.assertIn('"event": "status"', (root / "events.jsonl").read_text())

    def test_incomplete_status_is_ignored_and_checkpoint_is_published_atomically(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory); (root / "status.json").write_text("{incomplete")
            self.assertIsNone(read_latest(root))
            path = root / "model.pth"
            atomic_torch_save({"tensor": torch.ones(1)}, path)
            self.assertTrue(path.exists())
            self.assertFalse((root / "model.pth.tmp").exists())

    def test_status_replace_retries_transient_windows_sharing_error(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            target = Path(directory) / "status.json"; target.write_text('{"old": true}')
            transient = PermissionError(5, "Access denied"); transient.winerror = 5
            real_replace = os.replace
            attempts = 0
            def replace_once_then_succeed(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts == 1: raise transient
                return real_replace(source, destination)
            with patch("posenerf.telemetry.os.replace", side_effect=replace_once_then_succeed) as replace:
                atomic_write_json(target, {"new": True})
            self.assertEqual(replace.call_count, 2)
            self.assertEqual(json.loads(target.read_text()), {"new": True})
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_status_replace_preserves_existing_status_after_exhausted_sharing_errors(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            target = Path(directory) / "status.json"; target.write_text('{"old": true}')
            transient = PermissionError(5, "Access denied"); transient.winerror = 5
            with patch("posenerf.telemetry.os.replace", side_effect=transient), patch("posenerf.telemetry.time.sleep"):
                with self.assertRaises(PermissionError): atomic_write_json(target, {"new": True})
            self.assertEqual(json.loads(target.read_text()), {"old": True})
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_scene_snapshot_uses_camera_to_world_translation(self):
        image = torch.tensor([[[[1.0, 0.0, 0.0]]]])
        depth = torch.tensor([[[2.0]]]); mask = torch.ones_like(depth)
        pose = torch.eye(4).unsqueeze(0); pose[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
        directions = torch.tensor([[[0.0, 0.0, 1.0]]])
        snapshot = scene_snapshot(image, depth, mask, pose, directions, max_points=1)
        self.assertEqual(snapshot["points"][0]["position"], [1.0, 2.0, 5.0])
        self.assertEqual(snapshot["cameras"][0], [1.0, 2.0, 3.0])
        self.assertEqual(len(camera_frustum_vertices(pose[0])), 5)

    def test_predicted_snapshot_backprojects_predictions_and_records_provenance(self):
        pose = torch.eye(4).unsqueeze(0); pose[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
        rendered = [(0, torch.zeros(1, 1, 3), torch.tensor([[[0.0, 1.0, 0.0]]]), torch.tensor([[2.0]]))]
        directions = torch.tensor([[[0.0, 0.0, 1.0]]])
        snapshot = predicted_scene_snapshot(rendered, pose, directions, checkpoint_step=3000, near=0.1, far=4.5, max_points=1)
        self.assertEqual(snapshot["provenance"]["checkpoint_step"], 3000)
        self.assertIn("predicted depth only", snapshot["provenance"]["source"])
        self.assertEqual(snapshot["points"][0]["position"], [1.0, 2.0, 5.0])

    def test_dashboard_discovers_only_published_snapshot_pairs(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory); run = root / "b1"; live = run / "live"
            RunTelemetry(live).status("completed", 12, preview="visualizations/train_validation/hierarchical_nerf_val_00012.png", observed_scene="live/observed.json", nerf_scene="live/nerf_predicted_step_00012.json")
            preview = run / "visualizations" / "train_validation"; preview.mkdir(parents=True)
            (preview / "hierarchical_nerf_val_00012.png").write_bytes(b"png")
            (preview / "hierarchical_nerf_val_00013.png").write_bytes(b"orphan")
            (live / "nerf_predicted_step_00012.json").write_text("{}")
            (live / "observed.json").write_text("{}")
            (preview / "hierarchical_nerf_val_00012.json").write_text('{"average_psnr": 21.5, "average_depth_mae": 0.2}')
            self.assertEqual(_runs(root)[0]["name"], "b1")
            payload = _payload(root, "b1", "12")
            self.assertEqual(payload["selected_step"], 12)
            self.assertEqual(payload["snapshots"], [{"step": 12, "current": True}])
            self.assertEqual(payload["nerf_scene"], "b1/live/nerf_predicted_step_00012.json")
            self.assertEqual(payload["metrics"]["average_psnr"], 21.5)
            self.assertTrue(payload["history"])
            self.assertIsNone(_payload(root, "../outside", None))

    def test_dashboard_page_has_responsive_analytics_and_provenance_controls(self):
        for required in ("grid-template-columns:285px", "Training analytics", "NeRF-Predicted 3D Reconstruction", "data-preset", "onwheel"):
            self.assertIn(required, PAGE)
