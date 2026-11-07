import unittest

try:
    import torch
    from posenerf.training import build_sobel_edge_maps, sample_ray_batch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required")
class RaySamplerTests(unittest.TestCase):
    def test_cpu_sampler_is_deterministic_and_preserves_targets(self):
        images = torch.arange(2 * 3 * 4 * 3, dtype=torch.float32).reshape(2, 3, 4, 3)
        depths = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        masks = torch.ones(2, 3, 4)
        poses = torch.eye(4).repeat(2, 1, 1)
        directions = torch.stack(torch.meshgrid(torch.arange(3.), torch.arange(4.), indexing="ij"), dim=-1)
        directions = torch.cat([directions, torch.ones(3, 4, 1)], dim=-1)
        first = sample_ray_batch(images, depths, masks, poses, directions, 8, torch.device("cpu"), torch.Generator().manual_seed(7))
        second = sample_ray_batch(images, depths, masks, poses, directions, 8, torch.device("cpu"), torch.Generator().manual_seed(7))
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.isin(first[2].reshape(-1), images.reshape(-1)).all())

    def test_edge_mixture_is_deterministic_and_edge_biased(self):
        images = torch.zeros(2, 8, 8, 3); images[:, :, 4:] = 1.0
        depths = torch.ones(2, 8, 8); masks = torch.ones(2, 8, 8); masks[:, 0, 0] = 0
        poses = torch.eye(4).repeat(2, 1, 1)
        directions = torch.ones(8, 8, 3)
        edges = build_sobel_edge_maps(images, 0.2)
        first = sample_ray_batch(images, depths, masks, poses, directions, 1024, torch.device("cpu"), torch.Generator().manual_seed(9), edges, 0.3, True)
        second = sample_ray_batch(images, depths, masks, poses, directions, 1024, torch.device("cpu"), torch.Generator().manual_seed(9), edges, 0.3, True)
        for left, right in zip(first[:5], second[:5]):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(first[5], second[5])
        self.assertGreater(first[5]["edge_pixel_fraction"], float(edges.float().mean()))
        self.assertTrue(torch.isfinite(first[3]).all())

    def test_empty_edge_map_falls_back_to_exact_uniform_sampler(self):
        images = torch.zeros(1, 3, 4, 3); depths = torch.ones(1, 3, 4); masks = torch.ones(1, 3, 4)
        poses = torch.eye(4).unsqueeze(0); directions = torch.ones(3, 4, 3)
        uniform = sample_ray_batch(images, depths, masks, poses, directions, 16, torch.device("cpu"), torch.Generator().manual_seed(3))
        fallback = sample_ray_batch(images, depths, masks, poses, directions, 16, torch.device("cpu"), torch.Generator().manual_seed(3), torch.zeros(1, 3, 4, dtype=torch.bool), 0.3)
        for left, right in zip(uniform, fallback):
            self.assertTrue(torch.equal(left, right))
