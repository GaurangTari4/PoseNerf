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
from .utils import save_json
from .visualize import save_eval_figure


@torch.no_grad()
def build_ray_pool(images: torch.Tensor, depths: torch.Tensor, masks: torch.Tensor, poses: torch.Tensor, ray_dirs_cam: torch.Tensor) -> tuple[torch.Tensor, ...]:
    rays_o_all, rays_d_all, rgb_all, depth_all, mask_all = [], [], [], [], []
    for img, depth, mask, pose in tqdm(list(zip(images, depths, masks, poses)), desc="Building ray pool"):
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
def evaluate_model(model: HierarchicalNeRF, dataset: LoadedDataset, ray_dirs_cam: torch.Tensor, config: ExperimentConfig, frame_indices: list[int]) -> tuple[list[dict], list[tuple], float, float]:
    metrics = []
    rendered = []
    model.eval()
    for idx in frame_indices:
        pred_rgb, pred_depth = render_image(model, dataset.test_poses[idx], ray_dirs_cam, config)
        gt_rgb = dataset.test_images[idx]
        gt_depth = dataset.test_depths[idx]
        gt_mask = dataset.test_masks[idx] > 0.5
        psnr = -10.0 * math.log10(F.mse_loss(pred_rgb, gt_rgb).item())
        depth_mae = (pred_depth[gt_mask] - gt_depth[gt_mask]).abs().mean().item() if gt_mask.any() else float("nan")
        metrics.append({"index": idx, "psnr": psnr, "depth_mae": depth_mae})
        rendered.append((idx, gt_rgb, pred_rgb, pred_depth))
    avg_psnr = sum(m["psnr"] for m in metrics) / len(metrics)
    valid_depth_mae = [m["depth_mae"] for m in metrics if not math.isnan(m["depth_mae"])]
    avg_depth_mae = sum(valid_depth_mae) / len(valid_depth_mae) if valid_depth_mae else float("nan")
    return metrics, rendered, avg_psnr, avg_depth_mae


def save_evaluation_metrics(paths: PathsConfig, metrics: list[dict], avg_psnr: float, avg_depth_mae: float, filename: str = "evaluation_metrics.json") -> None:
    save_json(
        paths.visualizations_dir / filename,
        {
            "frames": metrics,
            "average_psnr": avg_psnr,
            "average_depth_mae": avg_depth_mae,
        },
    )


def train_model(model: HierarchicalNeRF, baseline_model: HierarchicalNeRF, dataset: LoadedDataset, ray_dirs_cam: torch.Tensor, config: ExperimentConfig, paths: PathsConfig, device: torch.device) -> Path:
    train_cfg = config.train
    train_rays_o, train_rays_d, train_rgb_pool, train_depth_pool, train_mask_pool = build_ray_pool(
        dataset.train_images, dataset.train_depths, dataset.train_masks, dataset.train_poses, ray_dirs_cam
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.num_steps, eta_min=train_cfg.lr * 0.2)
    scaler = GradScaler(device.type, enabled=(train_cfg.use_amp and device.type == "cuda"))
    loss_history = []
    val_dir = paths.visualizations_dir / "train_validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    resume_path = paths.checkpoints_dir / "hierarchical_nerf_resume_latest.pth"
    best_path = paths.checkpoints_dir / "hierarchical_nerf_best.pth"
    checkpoint_path = paths.checkpoints_dir / "hierarchical_nerf_trained.pth"
    start_step = 0
    val_frame_indices = list(range(min(train_cfg.val_frame_count, len(dataset.test_images))))
    current_signature = config.signature()
    best_val_psnr = float("-inf")
    best_val_step = -1
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    resumed = False

    if train_cfg.resume and resume_path.exists():
        resume_state = torch.load(resume_path, map_location=device)
        if resume_state.get("config_signature") == current_signature:
            model.load_state_dict(resume_state["model_state_dict"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            if resume_state.get("scaler_state_dict"):
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            loss_history = list(resume_state.get("loss_history", []))
            start_step = int(resume_state.get("step", -1)) + 1
            resumed = True
            print(f"Resumed training from step {start_step} using {resume_path}")
        else:
            print("Resume checkpoint config mismatch detected. Ignoring stale resume state and starting fresh.")
    else:
        print("Starting fresh HierarchicalNeRF training run.")

    bootstrap_metrics, rendered, bootstrap_avg_psnr, bootstrap_avg_depth_mae = evaluate_model(
        model, dataset, ray_dirs_cam, config, val_frame_indices
    )
    save_eval_figure(rendered, bootstrap_metrics, bootstrap_avg_psnr, bootstrap_avg_depth_mae, val_dir / "bootstrap_eval.png")
    save_evaluation_metrics(paths, bootstrap_metrics, bootstrap_avg_psnr, bootstrap_avg_depth_mae, "bootstrap_metrics.json")
    best_val_psnr = bootstrap_avg_psnr
    best_val_step = start_step - 1 if resumed else -1
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(
        {
            "step": best_val_step,
            "best_val_psnr": best_val_psnr,
            "best_val_step": best_val_step,
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "config_signature": current_signature,
            "val_metrics": bootstrap_metrics,
        },
        best_path,
    )

    model.train()
    for step in tqdm(range(start_step, train_cfg.num_steps), desc="Training HierarchicalNeRF"):
        idx = torch.randint(0, train_rays_o.shape[0], (train_cfg.batch_rays,), device=device)
        rays_o = train_rays_o[idx]
        rays_d = train_rays_d[idx]
        rgb_gt = train_rgb_pool[idx]
        depth_gt = train_depth_pool[idx]
        depth_mask = train_mask_pool[idx] > 0.5
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=(train_cfg.use_amp and device.type == "cuda")):
            rgb_pred, depth_pred, _ = render_rays(model, rays_o, rays_d, config, perturb=True)
            rgb_loss = F.mse_loss(rgb_pred, rgb_gt)
            depth_loss = F.smooth_l1_loss(depth_pred[depth_mask], depth_gt[depth_mask]) if depth_mask.any() else torch.zeros((), device=device)
            loss = train_cfg.rgb_loss_weight * rgb_loss + train_cfg.depth_loss_weight * depth_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss_history.append(loss.item())

        should_checkpoint = ((step + 1) % train_cfg.checkpoint_every == 0) or ((step + 1) == train_cfg.num_steps)
        should_validate = ((step + 1) % train_cfg.val_render_interval == 0) or ((step + 1) == train_cfg.num_steps)
        if should_validate:
            metrics, rendered, avg_psnr, avg_depth_mae = evaluate_model(model, dataset, ray_dirs_cam, config, val_frame_indices)
            save_eval_figure(rendered, metrics, avg_psnr, avg_depth_mae, val_dir / f"hierarchical_nerf_val_{step + 1:05d}.png")
            save_json(
                val_dir / f"hierarchical_nerf_val_{step + 1:05d}.json",
                {"frames": metrics, "average_psnr": avg_psnr, "average_depth_mae": avg_depth_mae},
            )
            if avg_psnr > best_val_psnr:
                best_val_psnr = avg_psnr
                best_val_step = step
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "step": step,
                        "best_val_psnr": best_val_psnr,
                        "best_val_step": best_val_step,
                        "model_state_dict": model.state_dict(),
                        "config": config.to_dict(),
                        "config_signature": current_signature,
                        "val_metrics": metrics,
                    },
                    best_path,
                )
            model.train()
        if should_checkpoint:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "best_model_state_dict": best_state_dict,
                    "best_val_psnr": best_val_psnr,
                    "best_val_step": best_val_step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss_history": loss_history,
                    "config": config.to_dict(),
                    "config_signature": current_signature,
                },
                resume_path,
            )

    model.load_state_dict(best_state_dict)
    torch.save(
        {
            "step": best_val_step if best_val_step >= 0 else train_cfg.num_steps - 1,
            "best_val_psnr": best_val_psnr,
            "best_val_step": best_val_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss_history": loss_history,
            "config": config.to_dict(),
            "config_signature": current_signature,
        },
        checkpoint_path,
    )
    baseline_model.load_state_dict(model.state_dict())
    baseline_model.eval()
    print(f"Saved trained checkpoint to: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model: HierarchicalNeRF, baseline_model: HierarchicalNeRF, checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    baseline_model.load_state_dict(model.state_dict())
    baseline_model.eval()
    model.eval()
    return checkpoint
