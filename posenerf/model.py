from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig


def get_ray_directions(config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    scene = config.scene
    ys, xs = torch.meshgrid(
        torch.arange(scene.height, dtype=torch.float32, device=device),
        torch.arange(scene.width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    if scene.use_opencv_camera_convention:
        return torch.stack([(xs - scene.cx) / scene.fx, (ys - scene.cy) / scene.fy, torch.ones_like(xs)], dim=-1)
    return torch.stack([(xs - scene.cx) / scene.fx, -(ys - scene.cy) / scene.fy, -torch.ones_like(xs)], dim=-1)


def get_rays(directions: torch.Tensor, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    rays_d = directions @ rotation.T
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = translation.expand_as(rays_d)
    return rays_o, rays_d


def positional_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    enc = [x]
    for i in range(num_freqs):
        freq = 2.0**i
        enc.append(torch.sin(freq * x))
        enc.append(torch.cos(freq * x))
    return torch.cat(enc, dim=-1)


class RadianceField(nn.Module):
    def __init__(self, pos_freqs: int, dir_freqs: int, hidden: int, num_layers: int) -> None:
        super().__init__()
        self.pos_freqs = pos_freqs
        self.dir_freqs = dir_freqs
        in_xyz = 3 + 3 * 2 * pos_freqs
        in_dir = 3 + 3 * 2 * dir_freqs
        self.skip_layer = num_layers // 2
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                inp = in_xyz
            elif i == self.skip_layer:
                inp = hidden + in_xyz
            else:
                inp = hidden
            self.layers.append(nn.Linear(inp, hidden))
        self.sigma_head = nn.Linear(hidden, 1)
        self.feature_head = nn.Linear(hidden, hidden)
        self.color_head = nn.Sequential(
            nn.Linear(hidden + in_dir, hidden // 2),
            nn.ReLU(True),
            nn.Linear(hidden // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, xyz: torch.Tensor, dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xyz_enc = positional_encoding(xyz, self.pos_freqs)
        dir_enc = positional_encoding(dirs, self.dir_freqs)
        h = xyz_enc
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                h = torch.cat([h, xyz_enc], dim=-1)
            h = F.relu(layer(h))
        # Softplus avoids a fully dead density field at initialization while
        # retaining non-negative volume densities.
        sigma = F.softplus(self.sigma_head(h))
        feat = self.feature_head(h)
        rgb = self.color_head(torch.cat([feat, dir_enc], dim=-1))
        return rgb, sigma


class HierarchicalNeRF(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        model_cfg = config.model
        self.coarse = RadianceField(model_cfg.pos_freqs, model_cfg.dir_freqs, model_cfg.hidden, model_cfg.layers)
        self.fine = RadianceField(model_cfg.pos_freqs, model_cfg.dir_freqs, model_cfg.hidden, model_cfg.layers)


def volume_render(rgb: torch.Tensor, sigma: torch.Tensor, z_vals: torch.Tensor) -> tuple[torch.Tensor, ...]:
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    deltas = torch.cat([deltas, 1e10 * torch.ones_like(deltas[:, :1])], dim=-1)
    alpha = 1.0 - torch.exp(-sigma * deltas)
    trans = torch.cumprod(
        torch.cat([torch.ones((z_vals.shape[0], 1), device=z_vals.device), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[:, :-1]
    weights = alpha * trans
    rgb_map = torch.sum(weights[..., None] * rgb, dim=1)
    depth_map = torch.sum(weights * z_vals, dim=1)
    acc_map = torch.sum(weights, dim=1)
    rgb_map = rgb_map + (1.0 - acc_map[..., None])
    return rgb_map, depth_map, acc_map, weights


def run_field(field: RadianceField, rays_o: torch.Tensor, rays_d: torch.Tensor, z_vals: torch.Tensor) -> tuple[torch.Tensor, ...]:
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
    view_dirs = rays_d[:, None, :].expand_as(pts)
    rgb, sigma = field(pts.reshape(-1, 3), view_dirs.reshape(-1, 3))
    rgb = rgb.reshape(z_vals.shape[0], z_vals.shape[1], 3)
    sigma = sigma.reshape(z_vals.shape[0], z_vals.shape[1])
    return volume_render(rgb, sigma, z_vals)


def sample_coarse_z_vals(rays_o: torch.Tensor, near: float, far: float, num_samples: int, perturb: bool) -> torch.Tensor:
    t_vals = torch.linspace(0.0, 1.0, num_samples, device=rays_o.device)
    z_vals = (near * (1.0 - t_vals) + far * t_vals).expand(rays_o.shape[0], num_samples)
    if perturb:
        mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)
    return z_vals


def sample_pdf(bins: torch.Tensor, weights: torch.Tensor, num_samples: int, deterministic: bool = False) -> torch.Tensor:
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
    if deterministic:
        u = torch.linspace(0.5 / num_samples, 1.0 - 0.5 / num_samples, num_samples, device=bins.device)
        u = u.expand(cdf.shape[0], num_samples).contiguous()
    else:
        u = torch.rand(cdf.shape[0], num_samples, device=bins.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0, max=bins.shape[-1] - 1)
    above = torch.clamp(inds, min=1, max=bins.shape[-1] - 1)
    cdf_below = torch.gather(cdf, 1, below)
    cdf_above = torch.gather(cdf, 1, above)
    bins_below = torch.gather(bins, 1, below)
    bins_above = torch.gather(bins, 1, above)
    denom = torch.where((cdf_above - cdf_below) < 1e-5, torch.ones_like(cdf_above), cdf_above - cdf_below)
    t = (u - cdf_below) / denom
    return bins_below + t * (bins_above - bins_below)


def render_rays(
    model: HierarchicalNeRF,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    config: ExperimentConfig,
    perturb: bool = True,
    num_coarse_samples: int | None = None,
    num_fine_samples: int | None = None,
    return_coarse: bool = False,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, ...]:
    train_cfg = config.train
    coarse_samples = num_coarse_samples if num_coarse_samples is not None else train_cfg.num_coarse_samples
    fine_samples = num_fine_samples if num_fine_samples is not None else train_cfg.num_fine_samples
    z_coarse = sample_coarse_z_vals(rays_o, config.scene.near, config.scene.far, coarse_samples, perturb)
    coarse_rgb, coarse_depth, coarse_acc, coarse_weights = run_field(model.coarse, rays_o, rays_d, z_coarse)
    if fine_samples <= 0:
        result = (coarse_rgb, coarse_depth, coarse_acc, coarse_rgb, coarse_depth, coarse_acc) if return_coarse else (coarse_rgb, coarse_depth, coarse_acc)
        return (*result, {"coarse_z_vals": z_coarse.detach(), "coarse_weights": coarse_weights.detach(), "fine_z_vals": z_coarse.detach(), "fine_weights": coarse_weights.detach()}) if return_diagnostics else result
    z_mids = 0.5 * (z_coarse[:, :-1] + z_coarse[:, 1:])
    z_fine = sample_pdf(z_mids, coarse_weights[:, 1:-1].detach(), fine_samples, deterministic=not perturb)
    z_all = torch.sort(torch.cat([z_coarse, z_fine], dim=-1), dim=-1).values
    fine_rgb, fine_depth, fine_acc, fine_weights = run_field(model.fine, rays_o, rays_d, z_all)
    diagnostics = {"coarse_z_vals": z_coarse.detach(), "coarse_weights": coarse_weights.detach(), "fine_z_vals": z_all.detach(), "fine_weights": fine_weights.detach()}
    if return_coarse:
        result = (fine_rgb, fine_depth, fine_acc, coarse_rgb, coarse_depth, coarse_acc)
    else:
        result = (fine_rgb, fine_depth, fine_acc)
    return (*result, diagnostics) if return_diagnostics else result


@torch.inference_mode()
def render_image(
    model: HierarchicalNeRF,
    c2w: torch.Tensor,
    ray_dirs_cam: torch.Tensor,
    config: ExperimentConfig,
    chunk: int | None = None,
    num_coarse_samples: int | None = None,
    num_fine_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rays_o, rays_d = get_rays(ray_dirs_cam, c2w)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    rgb_chunks, depth_chunks = [], []
    render_chunk = chunk if chunk is not None else config.model.render_chunk
    for start in range(0, rays_o.shape[0], render_chunk):
        end = start + render_chunk
        rgb, depth, _ = render_rays(
            model,
            rays_o[start:end],
            rays_d[start:end],
            config,
            perturb=False,
            num_coarse_samples=num_coarse_samples,
            num_fine_samples=num_fine_samples,
        )
        rgb_chunks.append(rgb)
        depth_chunks.append(depth)
    return (
        torch.cat(rgb_chunks, 0).reshape(config.scene.height, config.scene.width, 3).clamp(0.0, 1.0),
        torch.cat(depth_chunks, 0).reshape(config.scene.height, config.scene.width),
    )
