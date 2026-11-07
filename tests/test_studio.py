import unittest

import numpy as np

from posenerf.studio import CV_TO_THREE, IMPORT_MAP, QUALITY, STUDIO_NAV_PATCH, PAGE, three_to_posenerf_c2w
from posenerf.checkpoints import discover_checkpoints


class StudioCoordinateTests(unittest.TestCase):
    def test_three_camera_basis_round_trips_to_opencv_c2w(self):
        cv_pose = np.eye(4, dtype=np.float32); cv_pose[:3, 3] = [1.0, 2.0, 3.0]
        three_pose = cv_pose.copy(); three_pose[:3, :3] = cv_pose[:3, :3] @ CV_TO_THREE
        column_major_rows = three_pose.T.tolist()
        np.testing.assert_allclose(three_to_posenerf_c2w(column_major_rows), cv_pose, atol=1e-6)

    def test_invalid_pose_and_quality_are_rejected(self):
        with self.assertRaises(ValueError):
            three_to_posenerf_c2w([[0.0] * 4] * 4)
        self.assertEqual(QUALITY["fast"], (80, 60, 16, 16))

    def test_three_import_map_resolves_orbit_controls_dependency(self):
        self.assertIn('"three"', IMPORT_MAP)
        self.assertIn("three.module.js", IMPORT_MAP)

    def test_checkpoint_discovery_ignores_non_renderable_files(self):
        import tempfile
        from pathlib import Path
        import torch
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            d = Path(root) / "checkpoints"; d.mkdir()
            torch.save({"model_state_dict": {}, "config": {"train": {"num_steps": 12}}}, d / "hierarchical_nerf_trained.pth")
            torch.save({"optimizer": {}}, d / "hierarchical_nerf_resume_latest.pth")
            found = discover_checkpoints(Path(root), torch.device("cpu"))
            self.assertEqual([(x["kind"], x["step"]) for x in found], [("final", 12)])

    def test_navigation_ui_uses_stable_preset_ids_and_scene_points(self):
        """Keep the browser-control smoke paths independent of option order."""
        self.assertIn("p.id", STUDIO_NAV_PATCH)
        self.assertIn("ps.find(x=>x.id===selectedPresetId)", STUDIO_NAV_PATCH)
        self.assertIn("p.pose.flat()).transpose()", STUDIO_NAV_PATCH)
        self.assertIn("bridge.presetLocked=true", STUDIO_NAV_PATCH)
        self.assertIn("sceneBounds", STUDIO_NAV_PATCH)
        self.assertIn("PoseNeRFStudio.pred=pred", PAGE)
        self.assertIn("renameBookmark", STUDIO_NAV_PATCH)
        self.assertNotIn("geometry.boundingBox.constructor", STUDIO_NAV_PATCH)

    def test_browser_inspector_exposes_pose_and_render_metadata(self):
        """The UI smoke harness can compare visible pose/request metadata."""
        self.assertIn('id="poseMatrix"', PAGE)
        self.assertIn('id="renderMeta"', PAGE)
        self.assertIn("camera_pose:x.camera_pose", PAGE)
        self.assertIn("presetLocked", PAGE)
