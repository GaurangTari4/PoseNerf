from __future__ import annotations

import math

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
    return torch.stack(
        [
            torch.stack([torch.zeros_like(x), -z, y]),
            torch.stack([z, torch.zeros_like(x), -x]),
            torch.stack([-y, x, torch.zeros_like(x)]),
        ]
    )


def axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec) + 1e-9
    k = rotvec / theta
    K = skew(k)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def compose_pose_delta(base_pose: torch.Tensor, rotvec: torch.Tensor, transvec: torch.Tensor) -> torch.Tensor:
    delta = torch.eye(4, device=base_pose.device, dtype=base_pose.dtype)
    delta[:3, :3] = axis_angle_to_matrix(rotvec)
    delta[:3, 3] = transvec
    return base_pose @ delta


def rotation_error_deg(est_pose: torch.Tensor, gt_pose: torch.Tensor) -> float:
    r_rel = est_pose[:3, :3] @ gt_pose[:3, :3].T
    trace = torch.clamp((torch.trace(r_rel) - 1.0) * 0.5, -1.0, 1.0)
    return torch.rad2deg(torch.arccos(trace)).item()


def translation_error_m(est_pose: torch.Tensor, gt_pose: torch.Tensor) -> float:
    return torch.linalg.norm(est_pose[:3, 3] - gt_pose[:3, 3]).item()


def estimate_pose_for_index(model: HierarchicalNeRF, dataset: LoadedDataset, ray_dirs_cam: torch.Tensor, config: ExperimentConfig, device: torch.device, eval_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    pose_cfg = config.pose
    target_rgb = dataset.test_images[eval_idx]
    target_depth = dataset.test_depths[eval_idx]
    target_mask = dataset.test_masks[eval_idx]
    gt_pose = dataset.test_poses[eval_idx]
    init_pose = compose_pose_delta(
        gt_pose,
        torch.tensor([0.0, math.radians(pose_cfg.init_rot_perturb_deg), 0.0], device=device),
        pose_cfg.init_trans_tensor(device),
    )
    rot_delta = nn.Parameter(torch.zeros(3, device=device))
    trans_delta = nn.Parameter(torch.zeros(3, device=device))
    optimizer = torch.optim.Adam([rot_delta, trans_delta], lr=pose_cfg.lr)
    maybe_empty_cuda_cache(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    loss_history: list[float] = []
    for _ in tqdm(range(pose_cfg.iters), desc=f"Pose estimation frame {eval_idx}"):
        pose = compose_pose_delta(init_pose, rot_delta, trans_delta)
        ys = torch.randint(0, config.scene.height, (pose_cfg.batch_rays,), device=device)
        xs = torch.randint(0, config.scene.width, (pose_cfg.batch_rays,), device=device)
        dirs = ray_dirs_cam[ys, xs]
        rays_o, rays_d = get_rays(dirs, pose)
        rgb_gt = target_rgb[ys, xs]
        depth_gt = target_depth[ys, xs]
        mask = target_mask[ys, xs] > 0.5
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=(config.train.use_amp and device.type == "cuda")):
            rgb_pred, depth_pred, _ = render_rays(
                model,
                rays_o,
                rays_d,
                config,
                perturb=False,
                num_coarse_samples=pose_cfg.render_coarse_samples,
                num_fine_samples=pose_cfg.render_fine_samples,
            )
            rgb_loss = F.mse_loss(rgb_pred, rgb_gt)
            depth_loss = F.smooth_l1_loss(depth_pred[mask], depth_gt[mask]) if mask.any() else torch.zeros((), device=device)
            loss = pose_cfg.rgb_loss_weight * rgb_loss + pose_cfg.depth_loss_weight * depth_loss
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return init_pose.detach(), compose_pose_delta(init_pose, rot_delta, trans_delta).detach(), gt_pose.detach(), loss_history


def _pose_cache_paths(paths: PathsConfig) -> tuple:
    return paths.pose_dir / "pose_metrics.json", paths.pose_dir / "estimated_pose_{:02d}.txt", paths.pose_dir / "gt_pose_{:02d}.txt"


def load_cached_pose_summaries(dataset: LoadedDataset, config: ExperimentConfig, paths: PathsConfig, device: torch.device) -> list[dict] | None:
    metrics_path, est_template, gt_template = _pose_cache_paths(paths)
    cached = load_json(metrics_path)
    if not cached:
        return None
    summaries = []
    for item in cached.get("frames", []):
        idx = int(item["index"])
        est_path = Path(str(est_template).format(idx))
        gt_path = Path(str(gt_template).format(idx))
        if not est_path.exists() or not gt_path.exists():
            return None
        summaries.append(
            {
                "index": idx,
                "est_pose": torch.from_numpy(np.loadtxt(est_path).astype(np.float32)).to(device),
                "gt_pose": torch.from_numpy(np.loadtxt(gt_path).astype(np.float32)).to(device),
                "target_rgb": dataset.test_images[idx],
                "target_depth": dataset.test_depths[idx],
                "target_mask": dataset.test_masks[idx],
                "est_psnr": item.get("est_psnr"),
                "est_rot_err": item.get("est_rot_err"),
                "est_trans_err": item.get("est_trans_err"),
            }
        )
    return summaries


def run_pose_estimation(model: HierarchicalNeRF, dataset: LoadedDataset, ray_dirs_cam: torch.Tensor, config: ExperimentConfig, paths: PathsConfig, device: torch.device, use_cache: bool = True) -> list[dict]:
    if use_cache:
        cached = load_cached_pose_summaries(dataset, config, paths, device)
        if cached:
            print(f"Loaded {len(cached)} cached pose estimates from {paths.pose_dir}")
            return cached
    pose_eval_indices = list(range(min(config.pose.frame_count, len(dataset.test_images))))
    pose_summaries = []
    for eval_idx in pose_eval_indices:
        maybe_empty_cuda_cache(device)
        init_pose, est_pose, gt_pose, pose_loss_history = estimate_pose_for_index(
            model, dataset, ray_dirs_cam, config, device, eval_idx
        )
        with torch.no_grad():
            init_render, _ = render_image(
                model,
                init_pose,
                ray_dirs_cam,
                config,
                chunk=config.model.render_chunk,
                num_coarse_samples=config.pose.render_coarse_samples,
                num_fine_samples=config.pose.render_fine_samples,
            )
            est_render, est_depth = render_image(
                model,
                est_pose,
                ray_dirs_cam,
                config,
                chunk=config.model.render_chunk,
                num_coarse_samples=config.pose.render_coarse_samples,
                num_fine_samples=config.pose.render_fine_samples,
            )
        target_rgb = dataset.test_images[eval_idx]
        target_depth = dataset.test_depths[eval_idx]
        target_mask = dataset.test_masks[eval_idx]
        summary = {
            "index": eval_idx,
            "init_pose": init_pose,
            "est_pose": est_pose,
            "gt_pose": gt_pose,
            "loss_history": pose_loss_history,
            "init_render": init_render,
            "est_render": est_render,
            "est_depth": est_depth,
            "target_rgb": target_rgb,
            "target_depth": target_depth,
            "target_mask": target_mask,
            "init_rot_err": rotation_error_deg(init_pose, gt_pose),
            "init_trans_err": translation_error_m(init_pose, gt_pose),
            "est_rot_err": rotation_error_deg(est_pose, gt_pose),
            "est_trans_err": translation_error_m(est_pose, gt_pose),
            "init_psnr": -10.0 * math.log10(F.mse_loss(init_render, target_rgb).item()),
            "est_psnr": -10.0 * math.log10(F.mse_loss(est_render, target_rgb).item()),
        }
        pose_summaries.append(summary)

    save_pose_figure(pose_summaries, paths.pose_dir / "hierarchical_nerf_pose_estimation_multiview.png")
    save_loss_plot(pose_summaries, paths.pose_dir / "pose_optimization_loss_multiview.png")
    frames_summary = []
    for summary in pose_summaries:
        idx = summary["index"]
        np.savetxt(paths.pose_dir / f"gt_pose_{idx:02d}.txt", summary["gt_pose"].detach().cpu().numpy(), fmt="%.8f")
        np.savetxt(paths.pose_dir / f"estimated_pose_{idx:02d}.txt", summary["est_pose"].detach().cpu().numpy(), fmt="%.8f")
        frames_summary.append(
            {
                "index": idx,
                "init_psnr": summary["init_psnr"],
                "est_psnr": summary["est_psnr"],
                "init_rot_err": summary["init_rot_err"],
                "est_rot_err": summary["est_rot_err"],
                "init_trans_err": summary["init_trans_err"],
                "est_trans_err": summary["est_trans_err"],
            }
        )
    save_json(
        paths.pose_dir / "pose_metrics.json",
        {
            "frames": frames_summary,
            "average_est_psnr": sum(s["est_psnr"] for s in pose_summaries) / len(pose_summaries),
            "average_est_rot_err": sum(s["est_rot_err"] for s in pose_summaries) / len(pose_summaries),
            "average_est_trans_err": sum(s["est_trans_err"] for s in pose_summaries) / len(pose_summaries),
        },
    )
    return pose_summaries
