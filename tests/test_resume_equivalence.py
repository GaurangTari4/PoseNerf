import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

try:
    import torch
    from posenerf.config import ExperimentConfig, build_paths, ensure_directories
    from posenerf.data import LoadedDataset
    from posenerf.model import HierarchicalNeRF, get_ray_directions
    from posenerf.training import load_checkpoint, train_model
    from posenerf.utils import safe_torch_load, seed_everything
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class ResumeEquivalenceTests(unittest.TestCase):
    def _config(self):
        c = ExperimentConfig(seed=17)
        c.scene.orig_w = c.scene.orig_h = 4
        c.scene.orig_fx = c.scene.orig_fy = 2.0
        c.scene.orig_cx = c.scene.orig_cy = 2.0
        c.scene.downsample = 1
        c.model.hidden, c.model.layers, c.model.pos_freqs, c.model.dir_freqs = 8, 4, 2, 1
        c.train.num_steps, c.train.batch_rays, c.train.accumulation_steps = 4, 8, 4
        c.train.num_coarse_samples = c.train.num_fine_samples = 4
        c.train.val_render_interval = c.train.checkpoint_every = 2
        c.train.val_frame_count, c.train.use_amp, c.train.resume = 1, False, True
        return c

    def _dataset(self):
        images = torch.rand(2, 4, 4, 3)
        depths, masks = torch.ones(2, 4, 4), torch.ones(2, 4, 4)
        poses = torch.eye(4).repeat(2, 1, 1)
        return LoadedDataset([], [], [], images, depths, masks, poses, images[:1].clone(), depths[:1].clone(), masks[:1].clone(), poses[:1].clone(), images[:1].clone(), depths[:1].clone(), masks[:1].clone(), poses[:1].clone())

    def test_interrupted_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            config = self._config(); dataset = self._dataset(); directions = get_ray_directions(config, torch.device("cpu"))
            seed_everything(config.seed); reference = HierarchicalNeRF(config); baseline = HierarchicalNeRF(config); reference_path = build_paths(root / "reference"); ensure_directories(reference_path); train_model(reference, baseline, dataset, directions, config, reference_path, torch.device("cpu"))
            seed_everything(config.seed); resumed = HierarchicalNeRF(config); resumed_baseline = HierarchicalNeRF(config); resume_path = build_paths(root / "resume"); ensure_directories(resume_path); interrupted = replace(config); interrupted.train = replace(config.train, stop_after_step=2); train_model(resumed, resumed_baseline, dataset, directions, interrupted, resume_path, torch.device("cpu")); train_model(resumed, resumed_baseline, dataset, directions, config, resume_path, torch.device("cpu"))
            for left, right in zip(reference.parameters(), resumed.parameters()):
                self.assertTrue(torch.allclose(left, right, atol=1e-7, rtol=0.0))
            reference_state = safe_torch_load(reference_path.checkpoints_dir / "hierarchical_nerf_trained.pth", torch.device("cpu"))
            resumed_state = safe_torch_load(resume_path.checkpoints_dir / "hierarchical_nerf_trained.pth", torch.device("cpu"))
            self.assertEqual(reference_state["loss_history"], resumed_state["loss_history"])
            self.assertEqual(reference_state["scheduler_state_dict"], resumed_state["scheduler_state_dict"])
            self.assertEqual(reference_state["optimizer_state_dict"]["param_groups"], resumed_state["optimizer_state_dict"]["param_groups"])
            self.assertEqual(reference_state["best_val_step"], resumed_state["best_val_step"])
            self.assertEqual(reference_state["best_val_psnr"], resumed_state["best_val_psnr"])

    def test_accumulation_updates_optimizer_and_scheduler_once_per_global_step(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            config = self._config(); dataset = self._dataset(); directions = get_ray_directions(config, torch.device("cpu"))
            seed_everything(config.seed); model = HierarchicalNeRF(config); baseline = HierarchicalNeRF(config)
            paths = build_paths(root / "step_counts"); ensure_directories(paths)
            train_model(model, baseline, dataset, directions, config, paths, torch.device("cpu"))
            state = safe_torch_load(paths.checkpoints_dir / "hierarchical_nerf_trained.pth", torch.device("cpu"))
            self.assertEqual(state["scheduler_state_dict"]["last_epoch"], config.train.num_steps)
            self.assertEqual(state["optimizer_step_count"], config.train.num_steps)
            self.assertEqual(len(state["loss_history"]), config.train.num_steps)
            optimizer_steps = {int(item["step"].item()) for item in state["optimizer_state_dict"]["state"].values()}
            self.assertEqual(optimizer_steps, {config.train.num_steps})

    def test_scheduler_is_guarded_by_successful_amp_step(self):
        source = Path("posenerf/training.py").read_text(encoding="utf-8")
        self.assertIn("optimizer_stepped =", source)
        guarded = source.index("if optimizer_stepped:")
        self.assertLess(guarded, source.index("scheduler.step()", guarded))
