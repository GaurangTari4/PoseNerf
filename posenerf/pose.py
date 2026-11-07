from __future__ import annotations

import hashlib
import math
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from .config import ExperimentConfig, PathsConfig
from .data import LoadedDataset
from .model import HierarchicalNeRF, get_rays, render_image, render_rays
from .utils import load_json, maybe_empty_cuda_cache, save_json
from .visualize import save_loss_plot, save_pose_figure


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v[0], v[1], v[2]
    return torch.stack([torch.stack([torch.zeros_like(x), -z, y]), torch.stack([z, torch.zeros_like(x), -x]), torch.stack([-y, x, torch.zeros_like(x)])])


def axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec) + 1e-9
    K = skew(rotvec / theta)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def compose_pose_delta(base_pose: torch.Tensor, rotvec: torch.Tensor, transvec: torch.Tensor) -> torch.Tensor:
    delta = torch.eye(4, device=base_pose.device, dtype=base_pose.dtype)
    delta[:3, :3] = axis_angle_to_matrix(rotvec)
    delta[:3, 3] = transvec
    return base_pose @ delta


def rotation_error_deg(est_pose: torch.Tensor, gt_pose: torch.Tensor) -> float:
    relative = est_pose[:3, :3] @ gt_pose[:3, :3].T
    cosine = torch.clamp((torch.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return torch.rad2deg(torch.arccos(cosine)).item()


def translation_error_m(est_pose: torch.Tensor, gt_pose: torch.Tensor) -> float:
    return torch.linalg.norm(est_pose[:3, 3] - gt_pose[:3, 3]).item()


@torch.no_grad()
def retrieve_initial_pose(target_rgb: torch.Tensor, train_images: torch.Tensor, train_poses: torch.Tensor) -> tuple[torch.Tensor, int]:
    """RGB-only nearest-view initializer; it never reads test pose or depth."""
    errors = (train_images - target_rgb.unsqueeze(0)).square().mean(dim=(1, 2, 3))
    index = int(torch.argmin(errors).item())
    return train_poses[index].clone(), index


@torch.no_grad()
def sample_pose_pixels(target_rgb: torch.Tensor, count: int, generator: torch.Generator, mode: str = "uniform", informative_fraction: float = 0.30) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded query-RGB-only sampling; no pose/depth/model data is consulted."""
    h, w = target_rgb.shape[:2]
    if mode == "uniform":
        return torch.randint(h, (count,), device=target_rgb.device, generator=generator), torch.randint(w, (count,), device=target_rgb.device, generator=generator)
    if mode not in {"informative", "mixed"}: raise ValueError("unknown pose sampling mode")
    gray = target_rgb[..., 0] * .2126 + target_rgb[..., 1] * .7152 + target_rgb[..., 2] * .0722
    gx = F.pad(gray[:, 2:] - gray[:, :-2], (1, 1)); gy = F.pad(gray[2:, :] - gray[:-2, :], (0, 0, 1, 1))
    interest = torch.nonzero(torch.hypot(gx, gy).reshape(-1) >= torch.quantile(torch.hypot(gx, gy), .75), as_tuple=False).flatten()
    edge_count = count if mode == "informative" else int(round(count * informative_fraction))
    if interest.numel() == 0: edge_count = 0
    uniform_count = count - edge_count
    flat = torch.cat((torch.randint(h * w, (uniform_count,), device=target_rgb.device, generator=generator), interest[torch.randint(interest.numel(), (edge_count,), device=target_rgb.device, generator=generator)]))
    order = torch.randperm(count, device=target_rgb.device, generator=generator)
    flat = flat[order]; return torch.div(flat, w, rounding_mode="floor"), torch.remainder(flat, w)

@torch.no_grad()
def split_pose_rays(target_rgb: torch.Tensor, optimize_count: int, holdout_count: int, generator: torch.Generator, mode: str = "uniform") -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """A fixed, seeded, disjoint RGB-only optimize/holdout split."""
    h, w = target_rgb.shape[:2]
    if optimize_count + holdout_count > h * w: raise ValueError("ray split exceeds image pixels")
    # A permutation makes disjointness structural; interest ordering only
    # affects optimization, while the holdout stays uniformly representative.
    perm = torch.randperm(h * w, device=target_rgb.device, generator=generator)
    opt, hold = perm[:optimize_count], perm[optimize_count:optimize_count + holdout_count]
    if mode != "uniform":
        iy, ix = sample_pose_pixels(target_rgb, optimize_count, generator, mode)
        opt = torch.unique(iy * w + ix)
        if opt.numel() < optimize_count:
            extra = perm[~torch.isin(perm, opt)][:optimize_count - opt.numel()]; opt=torch.cat((opt,extra))
        opt=opt[:optimize_count]; hold=perm[~torch.isin(perm,opt)][:holdout_count]
    return (torch.div(opt,w,rounding_mode='floor'),torch.remainder(opt,w)), (torch.div(hold,w,rounding_mode='floor'),torch.remainder(hold,w))


def optimize_pose_from_rgb(model: HierarchicalNeRF, target_rgb: torch.Tensor, initial_pose: torch.Tensor, ray_dirs_cam: torch.Tensor, config: ExperimentConfig, device: torch.device, description: str, generator: torch.Generator | None = None, sampling_mode: str = "uniform", conservative: bool = False, fixed_pixels: tuple[torch.Tensor,torch.Tensor] | None = None) -> tuple[torch.Tensor, list[float]]:
    """iNeRF-style, pose-only photometric optimization against the RGB target."""
    pose_cfg = config.pose
    rot_delta, trans_delta = nn.Parameter(torch.zeros(3, device=device)), nn.Parameter(torch.zeros(3, device=device))
    optimizer = torch.optim.Adam([{'params':[rot_delta], 'lr':pose_cfg.lr * (.25 if conservative else 1)}, {'params':[trans_delta], 'lr':pose_cfg.lr * (.10 if conservative else 1)}])
    was_training = model.training
    parameter_states = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    history: list[float] = []
    try:
        for _ in tqdm(range(pose_cfg.iters), desc=description):
            pose = compose_pose_delta(initial_pose, rot_delta, trans_delta)
            ys, xs = fixed_pixels if fixed_pixels is not None else sample_pose_pixels(target_rgb, pose_cfg.batch_rays, generator, sampling_mode)
            rays_o, rays_d = get_rays(ray_dirs_cam[ys, xs], pose)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=(config.train.use_amp and device.type == "cuda")):
                rgb_pred, _, _ = render_rays(model, rays_o, rays_d, config, perturb=False, num_coarse_samples=pose_cfg.render_coarse_samples, num_fine_samples=pose_cfg.render_fine_samples)
                loss = pose_cfg.rgb_loss_weight * F.mse_loss(rgb_pred, target_rgb[ys, xs])
                if conservative: loss = loss + .01 * (rot_delta.square().sum() + trans_delta.square().sum())
            loss.backward()
            torch.nn.utils.clip_grad_norm_([rot_delta, trans_delta], 1.0)
            optimizer.step()
            if conservative:
                with torch.no_grad():
                    rot_delta.mul_(min(1., .15 / max(float(rot_delta.norm()), 1e-9))); trans_delta.mul_(min(1., .10 / max(float(trans_delta.norm()), 1e-9)))
            history.append(loss.item())
    finally:
        for parameter, state in zip(model.parameters(), parameter_states):
            parameter.requires_grad_(state)
        model.train(was_training)
    return compose_pose_delta(initial_pose, rot_delta, trans_delta).detach(), history


def _cache_provenance(dataset: LoadedDataset, config: ExperimentConfig, checkpoint_id: str) -> str:
    payload = {"checkpoint": checkpoint_id, "pose": asdict(config.pose), "train": [str(frame.color_path) for frame in dataset.train_frames], "test": [str(frame.color_path) for frame in dataset.test_frames]}
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def load_cached_pose_summaries(dataset: LoadedDataset, paths: PathsConfig, device: torch.device, provenance: str) -> list[dict] | None:
    cached = load_json(paths.pose_dir / "pose_metrics.json")
    if not cached or cached.get("provenance") != provenance:
        return None
    summaries = []
    for item in cached.get("frames", []):
        idx = int(item["index"])
        pose_path = paths.pose_dir / f"estimated_pose_{idx:02d}.txt"
        if not pose_path.exists() or idx >= len(dataset.test_images):
            return None
        summaries.append({"index": idx, "est_pose": torch.from_numpy(np.loadtxt(pose_path).astype(np.float32)).to(device), "retrieval_index": item["retrieval_index"]})
    return summaries or None


def run_pose_estimation(model: HierarchicalNeRF, dataset: LoadedDataset, ray_dirs_cam: torch.Tensor, config: ExperimentConfig, paths: PathsConfig, device: torch.device, checkpoint_id: str, use_cache: bool = True) -> list[dict]:
    provenance = _cache_provenance(dataset, config, checkpoint_id)
    raw_summaries = load_cached_pose_summaries(dataset, paths, device, provenance) if use_cache else None
    if raw_summaries is None:
        raw_summaries = []
        for index in range(min(config.pose.frame_count, len(dataset.test_images))):
            target_rgb = dataset.test_images[index]
            init_pose, retrieval_index = retrieve_initial_pose(target_rgb, dataset.train_images, dataset.train_poses)
            maybe_empty_cuda_cache(device)
            est_pose, history = optimize_pose_from_rgb(model, target_rgb, init_pose, ray_dirs_cam, config, device, f"RGB pose optimization frame {index}")
            raw_summaries.append({"index": index, "init_pose": init_pose, "est_pose": est_pose, "retrieval_index": retrieval_index, "loss_history": history})

    summaries = []
    for raw in raw_summaries:
        index = raw["index"]
        target_rgb, gt_pose = dataset.test_images[index], dataset.test_poses[index]
        init_pose = raw.get("init_pose", dataset.train_poses[raw["retrieval_index"]].clone())
        with torch.no_grad():
            init_render, _ = render_image(model, init_pose, ray_dirs_cam, config, num_coarse_samples=config.pose.render_coarse_samples, num_fine_samples=config.pose.render_fine_samples)
            est_render, est_depth = render_image(model, raw["est_pose"], ray_dirs_cam, config, num_coarse_samples=config.pose.render_coarse_samples, num_fine_samples=config.pose.render_fine_samples)
        summaries.append({**raw, "init_pose": init_pose, "gt_pose": gt_pose, "target_rgb": target_rgb, "init_render": init_render, "est_render": est_render, "est_depth": est_depth, "init_rot_err": rotation_error_deg(init_pose, gt_pose), "init_trans_err": translation_error_m(init_pose, gt_pose), "est_rot_err": rotation_error_deg(raw["est_pose"], gt_pose), "est_trans_err": translation_error_m(raw["est_pose"], gt_pose), "init_psnr": -10.0 * math.log10(max(F.mse_loss(init_render, target_rgb).item(), 1e-12)), "est_psnr": -10.0 * math.log10(max(F.mse_loss(est_render, target_rgb).item(), 1e-12)), "loss_history": raw.get("loss_history", [])})

    save_pose_figure(summaries, paths.pose_dir / "rgb_localization_multiview.png")
    save_loss_plot(summaries, paths.pose_dir / "rgb_localization_loss_multiview.png")
    frames = []
    for summary in summaries:
        index = summary["index"]
        np.savetxt(paths.pose_dir / f"estimated_pose_{index:02d}.txt", summary["est_pose"].detach().cpu().numpy(), fmt="%.8f")
        frames.append({key: summary[key] for key in ("index", "retrieval_index", "init_psnr", "est_psnr", "init_rot_err", "est_rot_err", "init_trans_err", "est_trans_err")})
    save_json(paths.pose_dir / "pose_metrics.json", {"provenance": provenance, "inference_mode": "rgb_retrieval_plus_photometric_refinement", "frames": frames, "average_est_psnr": sum(s["est_psnr"] for s in summaries) / len(summaries), "average_est_rot_err": sum(s["est_rot_err"] for s in summaries) / len(summaries), "average_est_trans_err": sum(s["est_trans_err"] for s in summaries) / len(summaries)})
    return summaries
