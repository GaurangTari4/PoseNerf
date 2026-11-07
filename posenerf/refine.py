from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .config import ExperimentConfig, PathsConfig
from .data import LoadedDataset
from .model import HierarchicalNeRF, get_rays, render_image, render_rays
from .utils import safe_torch_load, save_json
from .visualize import save_refinement_figure


@torch.no_grad()
def build_refined_ray_pool(images: torch.Tensor, depths: torch.Tensor, masks: torch.Tensor, poses: torch.Tensor, ray_dirs_cam: torch.Tensor) -> tuple[torch.Tensor, ...]:
    rays_o_all, rays_d_all, rgb_all, depth_all, mask_all = [], [], [], [], []
    for img, depth, mask, pose in tqdm(list(zip(images, depths, masks, poses)), desc="Building refined ray pool"):
        rays_o, rays_d = get_rays(ray_dirs_cam, pose)
        rays_o_all.append(rays_o.reshape(-1, 3))
        rays_d_all.append(rays_d.reshape(-1, 3))
        rgb_all.append(img.reshape(-1, 3))
        depth_all.append(depth.reshape(-1))
        mask_all.append(mask.reshape(-1))
    return (
        torch.cat(rays_o_all, 0),
        torch.cat(rays_d_all, 0),
        torch.cat(rgb_all, 0),
        torch.cat(depth_all, 0),
        torch.cat(mask_all, 0),
    )


@torch.no_grad()
def evaluate_refinement_views(model: HierarchicalNeRF, dataset: LoadedDataset, ray_dirs_cam: torch.Tensor, config: ExperimentConfig, frame_indices: list[int]) -> tuple[list[dict], float, float]:
    stats = []
    model.eval()
    for idx in frame_indices:
        pred_rgb, pred_depth = render_image(model, dataset.test_poses[idx], ray_dirs_cam, config)
        gt_rgb = dataset.test_images[idx]
        gt_depth = dataset.test_depths[idx]
        gt_mask = dataset.test_masks[idx] > 0.5
        psnr = -10.0 * math.log10(F.mse_loss(pred_rgb, gt_rgb).item())
        depth_mae = (pred_depth[gt_mask] - gt_depth[gt_mask]).abs().mean().item() if gt_mask.any() else float("nan")
        stats.append({"index": idx, "pred_rgb": pred_rgb, "pred_depth": pred_depth, "gt_rgb": gt_rgb, "psnr": psnr, "depth_mae": depth_mae})
    avg_psnr = sum(s["psnr"] for s in stats) / len(stats)
    valid_depth_mae = [s["depth_mae"] for s in stats if not math.isnan(s["depth_mae"])]
    avg_depth_mae = sum(valid_depth_mae) / len(valid_depth_mae) if valid_depth_mae else float("nan")
    return stats, avg_psnr, avg_depth_mae


def run_refinement(model: HierarchicalNeRF, baseline_model: HierarchicalNeRF, dataset: LoadedDataset, ray_dirs_cam: torch.Tensor, pose_summaries: list[dict], config: ExperimentConfig, paths: PathsConfig, device: torch.device) -> Path:
    refine_cfg = config.refine
    refine_eval_indices = list(range(min(refine_cfg.frame_count, len(dataset.test_images))))
    refine_train_count = min(refine_cfg.train_images, len(dataset.train_images))
    refine_images = dataset.train_images[:refine_train_count]
    refine_depths = dataset.train_depths[:refine_train_count]
    refine_masks = dataset.train_masks[:refine_train_count]
    refine_poses = dataset.train_poses[:refine_train_count].clone()
    train_rays_o, train_rays_d, train_rgb_pool, train_depth_pool, train_mask_pool = build_refined_ray_pool(
        refine_images, refine_depths, refine_masks, refine_poses, ray_dirs_cam
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.load_state_dict(baseline_model.state_dict())
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=refine_cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=refine_cfg.steps, eta_min=refine_cfg.lr * 0.2)
    scaler = GradScaler(device.type, enabled=(config.train.use_amp and device.type == "cuda"))
    refine_resume_path = paths.checkpoints_dir / "hierarchical_nerf_refine_latest.pth"
    refine_best_path = paths.checkpoints_dir / "hierarchical_nerf_refine_best.pth"
    refined_ckpt = paths.checkpoints_dir / "hierarchical_nerf_refined.pth"
    refine_data_config = {
        "refine_train_count": refine_train_count,
        "refine_pose_count": 0,
        "pose_eval_indices": [],
    }
    refine_start_step = 0
    # Select refinement checkpoints on validation frames only. Final test evaluation happens in the CLI.
    validation_dataset = LoadedDataset(
        train_frames=dataset.train_frames, val_frames=dataset.val_frames, test_frames=dataset.test_frames,
        train_images=dataset.train_images, train_depths=dataset.train_depths, train_masks=dataset.train_masks, train_poses=dataset.train_poses,
        val_images=dataset.val_images, val_depths=dataset.val_depths, val_masks=dataset.val_masks, val_poses=dataset.val_poses,
        test_images=dataset.val_images, test_depths=dataset.val_depths, test_masks=dataset.val_masks, test_poses=dataset.val_poses,
    )
    refine_eval_indices = list(range(min(refine_cfg.frame_count, len(dataset.val_images))))
    before_stats, before_psnr, before_depth_mae = evaluate_refinement_views(baseline_model, validation_dataset, ray_dirs_cam, config, refine_eval_indices)
    best_psnr = before_psnr
    best_step = -1
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if refine_cfg.resume and refine_resume_path.exists():
        refine_state = safe_torch_load(refine_resume_path, device)
        if refine_state.get("refine_data_config") == refine_data_config:
            model.load_state_dict(refine_state["model_state_dict"])
            optimizer.load_state_dict(refine_state["optimizer_state_dict"])
            scheduler.load_state_dict(refine_state["scheduler_state_dict"])
            if refine_state.get("scaler_state_dict"):
                scaler.load_state_dict(refine_state["scaler_state_dict"])
            refine_start_step = int(refine_state.get("step", -1)) + 1
            best_psnr = float(refine_state.get("best_psnr", before_psnr))
            best_step = int(refine_state.get("best_step", -1))
            if "best_model_state_dict" in refine_state:
                best_state_dict = {k: v.detach().cpu().clone() for k, v in refine_state["best_model_state_dict"].items()}

    for step in tqdm(range(refine_start_step, refine_cfg.steps), desc="Conservative refinement"):
        idx = torch.randint(0, train_rays_o.shape[0], (refine_cfg.batch_rays,), device=device)
        rays_o = train_rays_o[idx]
        rays_d = train_rays_d[idx]
        rgb_gt = train_rgb_pool[idx]
        depth_gt = train_depth_pool[idx]
        depth_mask = train_mask_pool[idx] > 0.5
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=(config.train.use_amp and device.type == "cuda")):
            rgb_pred, depth_pred, _ = render_rays(model, rays_o, rays_d, config, perturb=True)
            rgb_loss = F.mse_loss(rgb_pred, rgb_gt)
            depth_loss = F.smooth_l1_loss(depth_pred[depth_mask], depth_gt[depth_mask]) if depth_mask.any() else torch.zeros((), device=device)
            loss = rgb_loss + refine_cfg.depth_loss_weight * depth_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        should_eval = (((step + 1) % refine_cfg.eval_every) == 0) or ((step + 1) == refine_cfg.steps)
        if should_eval:
            eval_stats, eval_psnr, eval_depth_mae = evaluate_refinement_views(model, validation_dataset, ray_dirs_cam, config, refine_eval_indices)
            if eval_psnr > best_psnr:
                best_psnr = eval_psnr
                best_step = step
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "step": step,
                        "best_psnr": best_psnr,
                        "best_step": best_step,
                        "model_state_dict": model.state_dict(),
                        "config": config.to_dict(),
                        "refine_data_config": refine_data_config,
                        "eval_stats": [{"index": s["index"], "psnr": s["psnr"], "depth_mae": s["depth_mae"]} for s in eval_stats],
                    },
                    refine_best_path,
                )
            model.train()
        if (((step + 1) % refine_cfg.checkpoint_every) == 0) or ((step + 1) == refine_cfg.steps):
            torch.save(
                {
                    "step": step,
                    "best_psnr": best_psnr,
                    "best_step": best_step,
                    "best_model_state_dict": best_state_dict,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": config.to_dict(),
                    "refine_data_config": refine_data_config,
                },
                refine_resume_path,
            )
    model.load_state_dict(best_state_dict)
    model.eval()
    after_stats, after_psnr, after_depth_mae = evaluate_refinement_views(model, validation_dataset, ray_dirs_cam, config, refine_eval_indices)
    save_refinement_figure(before_stats, after_stats, before_psnr, after_psnr, after_depth_mae, paths.refinement_dir / "hierarchical_nerf_refinement_multiview.png")
    save_json(
        paths.refinement_dir / "refinement_metrics.json",
        {
            "best_step": best_step + 1 if best_step >= 0 else None,
            "before_average_psnr": before_psnr,
            "after_average_psnr": after_psnr,
            "after_average_depth_mae": after_depth_mae,
            "frames": [
                {
                    "index": stats["index"],
                    "before_psnr": before_stats[i]["psnr"],
                    "after_psnr": stats["psnr"],
                    "after_depth_mae": stats["depth_mae"],
                }
                for i, stats in enumerate(after_stats)
            ],
        },
    )
    torch.save(
        {
            "step": best_step,
            "best_psnr": best_psnr,
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "refine_data_config": refine_data_config,
        },
        refined_ckpt,
    )
    return refined_ckpt
