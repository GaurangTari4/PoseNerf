import unittest

try:
    import torch
    from posenerf.config import ExperimentConfig
    from posenerf.model import HierarchicalNeRF, get_ray_directions, get_rays, render_rays
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required for NeRF gradient tests")
class HierarchicalGradientTests(unittest.TestCase):
    def test_identity_camera_center_ray_uses_opencv_forward_axis(self):
        config = ExperimentConfig()
        directions = get_ray_directions(config, torch.device("cpu"))
        origins, rays = get_rays(directions[config.scene.height // 2, config.scene.width // 2], torch.eye(4))
        self.assertTrue(torch.allclose(origins, torch.zeros(3)))
        self.assertTrue(torch.allclose(rays, torch.tensor([0.0, 0.0, 1.0])))

    def test_render_image_is_chunk_invariant(self):
        from posenerf.model import render_image
        config = ExperimentConfig()
        config.scene.downsample = 16
        config.model.hidden, config.model.layers = 16, 4
        config.model.pos_freqs, config.model.dir_freqs = 2, 1
        config.train.num_coarse_samples = config.train.num_fine_samples = 8
        model = HierarchicalNeRF(config).eval()
        directions = get_ray_directions(config, torch.device("cpu"))
        outputs = [render_image(model, torch.eye(4), directions, config, chunk=size) for size in (256, 512, 1024)]
        for rgb, depth in outputs[1:]:
            self.assertTrue(torch.allclose(outputs[0][0], rgb, atol=1e-6))
            self.assertTrue(torch.allclose(outputs[0][1], depth, atol=1e-6))

    def test_coarse_field_receives_supervised_gradient(self):
        config = ExperimentConfig()
        config.model.hidden = 16
        config.model.layers = 4
        config.model.pos_freqs = 2
        config.model.dir_freqs = 1
        config.train.num_coarse_samples = 8
        config.train.num_fine_samples = 8
        model = HierarchicalNeRF(config)
        rays_o = torch.zeros(6, 3)
        rays_d = torch.nn.functional.normalize(torch.randn(6, 3), dim=-1)
        target_rgb, target_depth = torch.rand(6, 3), torch.rand(6)
        fine_rgb, fine_depth, _, coarse_rgb, coarse_depth, _ = render_rays(model, rays_o, rays_d, config, return_coarse=True)
        loss = (fine_rgb - target_rgb).square().mean() + (fine_depth - target_depth).square().mean()
        loss = loss + (coarse_rgb - target_rgb).square().mean() + (coarse_depth - target_depth).square().mean()
        loss.backward()
        coarse_gradient_norm = sum(parameter.grad.abs().sum().item() for parameter in model.coarse.parameters() if parameter.grad is not None)
        fine_gradient_norm = sum(parameter.grad.abs().sum().item() for parameter in model.fine.parameters() if parameter.grad is not None)
        self.assertGreater(coarse_gradient_norm, 0.0)
        self.assertGreater(fine_gradient_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
