import inspect
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    from posenerf.config import ExperimentConfig
    from posenerf.data import load_depth, load_pose
    from posenerf.model import HierarchicalNeRF
    from posenerf.pose import optimize_pose_from_rgb
    from posenerf.training import load_checkpoint
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "Project runtime dependencies are required")
class CheckpointAndPoseSafetyTests(unittest.TestCase):
    def test_load_pose_rejects_non_homogeneous_matrix(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "invalid.pose.txt"
            np.savetxt(path, np.eye(3, dtype=np.float32))
            with self.assertRaises(ValueError):
                load_pose(path)

    def test_depth_loader_masks_zero_and_65535_values(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "depth.png"
            from PIL import Image
            Image.fromarray(np.array([[0, 1000], [65535, 2500]], dtype=np.uint16)).save(path)
            depth, mask = load_depth(path, downsample=1)
        self.assertTrue(np.array_equal(mask, np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)))
        self.assertTrue(np.allclose(depth, np.array([[0.0, 1.0], [0.0, 2.5]], dtype=np.float32)))

    def test_checkpoint_round_trip_requires_matching_signature(self):
        config = ExperimentConfig()
        config.model.hidden, config.model.layers = 16, 4
        config.model.pos_freqs, config.model.dir_freqs = 2, 1
        source, target, baseline = HierarchicalNeRF(config), HierarchicalNeRF(config), HierarchicalNeRF(config)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            checkpoint_path = Path(directory) / "model.pth"
            torch.save({"model_state_dict": source.state_dict(), "config_signature": config.signature()}, checkpoint_path)
            load_checkpoint(target, baseline, checkpoint_path, torch.device("cpu"), config)
        for expected, actual in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_rgb_optimizer_source_has_no_test_label_access(self):
        source = inspect.getsource(optimize_pose_from_rgb)
        self.assertNotIn("test_poses", source)
        self.assertNotIn("test_depth", source)
        self.assertNotIn("test_masks", source)


if __name__ == "__main__":
    unittest.main()
