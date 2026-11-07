import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

try:
    from posenerf.config import ExperimentConfig
    from posenerf.milestone9 import backproject_pixels, build_reference_database, c2w_points, project_world, camera_matrix, pnp_ransac, valid_rigid_c2w
except ModuleNotFoundError:
    backproject_pixels = None


@unittest.skipUnless(backproject_pixels is not None, "geometry dependencies required")
class GeometryTests(unittest.TestCase):
    def test_backproject_world_project_roundtrip(self):
        config = ExperimentConfig(); config.scene.downsample = 4; K = camera_matrix(config)
        pixels = np.array([[80.0, 60.0], [10.5, 40.25], [145.2, 85.4]]); depths = np.array([1.0, 2.0, 1.4])
        pose = np.eye(4); pose[:3, 3] = [.3, -.2, .7]
        world = c2w_points(backproject_pixels(pixels, depths, K), pose); reproj, z = project_world(world, pose, K)
        np.testing.assert_allclose(reproj, pixels, atol=1e-8); np.testing.assert_allclose(z, depths); self.assertTrue(valid_rigid_c2w(pose))

    def test_intrinsics_and_invalid_depth(self):
        config = ExperimentConfig(); config.scene.downsample = 4; K = camera_matrix(config)
        self.assertEqual(K[0, 0], 146.25); self.assertEqual(K[0, 2], 80.0)
        with self.assertRaises(ValueError): backproject_pixels(np.array([[1., 1.]]), np.array([0.]), K)

    def test_ray_distance_converts_to_camera_axis_depth(self):
        # A normalized ray through (fx, 0) has z component 1/sqrt(2).
        ray_distance = 2.0; camera_unit_z = 1.0 / np.sqrt(2.0)
        self.assertAlmostEqual(ray_distance * camera_unit_z, np.sqrt(2.0), places=7)

    def test_pnp_synthetic_and_failure(self):
        K = np.array([[200., 0., 80.], [0., 200., 60.], [0., 0., 1.]])
        rng = np.random.default_rng(7); world = rng.uniform([-1, -1, 2], [1, 1, 4], (30, 3))
        c2w = np.eye(4); c2w[:3, 3] = [.1, -.2, .3]; pixels, _ = project_world(world, c2w, K)
        result = pnp_ransac(pixels, world, K); self.assertTrue(result["ok"]); self.assertTrue(valid_rigid_c2w(result["pose"])); np.testing.assert_allclose(result["pose"], c2w, atol=1e-3)
        self.assertFalse(pnp_ransac(pixels[:4], world[:4], K)["ok"])

    def test_reference_validation_overlap_is_rejected_before_feature_extraction(self):
        scene = Path("scene"); frame = SimpleNamespace(color_path=scene / "seq-01" / "frame-000000.color.png")
        data = SimpleNamespace(train_frames=[frame], val_frames=[frame])
        with self.assertRaisesRegex(ValueError, "leaked"):
            build_reference_database(data, ExperimentConfig(), scene, Path("unused-cache"))


if __name__ == "__main__": unittest.main()
