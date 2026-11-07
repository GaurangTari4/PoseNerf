from __future__ import annotations

import hashlib
import math
import time
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .config import ExperimentConfig, PathsConfig
from .data import LoadedDataset
from .model import HierarchicalNeRF, get_rays, render_image, render_rays
from .utils import capture_rng_state, restore_rng_state, safe_torch_load, save_json
from .telemetry import RunTelemetry, atomic_torch_save, atomic_write_json


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_immutable_snapshot(checkpoints_dir: Path, snapshot: Path) -> None:
    """Append a verified immutable snapshot entry without ever replacing one."""
    manifest = checkpoints_dir / "immutable_snapshot_manifest.json"
    entries: dict[str, dict[str, str]] = {}
    if manifest.exists():
        import json
        entries = json.loads(manifest.read_text(encoding="utf-8")).get("snapshots", {})
    if snapshot.name in entries:
        raise FileExistsError(f"immutable snapshot is already recorded: {snapshot}")
    entries[snapshot.name] = {"sha256": _checkpoint_sha256(snapshot)}
    atomic_write_json(manifest, {"snapshots": entries})
    atomic_write_json(checkpoints_dir.parent / "e8_logs" / "checkpoint_manifest.json", {"snapshots": entries})


def _require_checkpoint_disk_space(path: Path, minimum_free_bytes: int = 2 * 1024**3) -> None:
    free = shutil.disk_usage(path.parent).free
    if free < minimum_free_bytes:
        raise RuntimeError(f"Insufficient free disk space for checkpointing: {free / 1024**3:.2f} GiB available")
from .scene import predicted_scene_snapshot, scene_snapshot
from .visualize import save_eval_figure


@torch.no_grad()
def build_sobel_edge_maps(images: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return deterministic per-training-image Sobel maps on CPU.

    Magnitudes are normalized independently per image so the fixed threshold
    is stable across exposure changes.  This intentionally accepts only the
    tensor supplied by the training split; validation/test frames never enter
    this routine.
    """
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must have shape [frames, height, width, 3]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("edge threshold must be in [0, 1]")
    gray = images.float().permute(0, 3, 1, 2)
    gray = 0.2126 * gray[:, :1] + 0.7152 * gray[:, 1:2] + 0.0722 * gray[:, 2:3]
    sobel_x = torch.tensor(((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0))).reshape(1, 1, 3, 3)
    magnitude = torch.hypot(F.conv2d(gray, sobel_x, padding=1), F.conv2d(gray, sobel_y, padding=1)).squeeze(1)
    scale = magnitude.amax(dim=(1, 2), keepdim=True).clamp_min(torch.finfo(magnitude.dtype).eps)
    return magnitude.div(scale).ge(threshold)


def _sample_indices(images: torch.Tensor, batch_rays: int, generator: torch.Generator, edge_maps: torch.Tensor | None, edge_fraction: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Sample CPU frame/pixel indices, retaining legacy draws for uniform mode."""
    if edge_maps is None or edge_fraction <= 0.0:
        # Keep this branch exactly equivalent to the historical sampler.
        frame_idx = torch.randint(len(images), (batch_rays,), generator=generator, device="cpu")
        ys = torch.randint(images.shape[1], (batch_rays,), generator=generator, device="cpu")
        xs = torch.randint(images.shape[2], (batch_rays,), generator=generator, device="cpu")
        return frame_idx, ys, xs, 0.0
    if not 0.0 <= edge_fraction <= 1.0:
        raise ValueError("edge sampling fraction must be in [0, 1]")
    if tuple(edge_maps.shape) != tuple(images.shape[:3]):
        raise ValueError("edge maps must match [frames, height, width]")
    edge_indices = torch.nonzero(edge_maps.reshape(-1), as_tuple=False).flatten()
    if edge_indices.numel() == 0:
        return _sample_indices(images, batch_rays, generator, None, 0.0)
    edge_count = int(round(batch_rays * edge_fraction))
    uniform_count = batch_rays - edge_count
    frame_idx = torch.randint(len(images), (uniform_count,), generator=generator, device="cpu")
    ys = torch.randint(images.shape[1], (uniform_count,), generator=generator, device="cpu")
    xs = torch.randint(images.shape[2], (uniform_count,), generator=generator, device="cpu")
    chosen = edge_indices[torch.randint(edge_indices.numel(), (edge_count,), generator=generator, device="cpu")]
    pixels_per_frame = images.shape[1] * images.shape[2]
    edge_frame = torch.div(chosen, pixels_per_frame, rounding_mode="floor")
    edge_pixel = torch.remainder(chosen, pixels_per_frame)
    frame_idx = torch.cat((frame_idx, edge_frame))
    ys = torch.cat((ys, torch.div(edge_pixel, images.shape[2], rounding_mode="floor")))
    xs = torch.cat((xs, torch.remainder(edge_pixel, images.shape[2])))
    order = torch.randperm(batch_rays, generator=generator, device="cpu")
    frame_idx, ys, xs = frame_idx[order], ys[order], xs[order]
    actual_edge_fraction = float(edge_maps[frame_idx, ys, xs].float().mean().item())
    return frame_idx, ys, xs, actual_edge_fraction


def sample_ray_batch(images: torch.Tensor, depths: torch.Tensor, masks: torch.Tensor, poses: torch.Tensor, ray_dirs_cam: torch.Tensor, batch_rays: int, device: torch.device, generator: torch.Generator, edge_maps: torch.Tensor | None = None, edge_fraction: float = 0.0, return_sampling_stats: bool = False) -> tuple[torch.Tensor, ...]:
    """Sample CPU-backed frames/pixels and transfer only the ray minibatch."""
    frame_idx, ys, xs, actual_edge_fraction = _sample_indices(images, batch_rays, generator, edge_maps, edge_fraction)
    rgb = images[frame_idx, ys, xs].to(device, non_blocking=True)
    depth = depths[frame_idx, ys, xs].to(device, non_blocking=True)
    mask = masks[frame_idx, ys, xs].to(device, non_blocking=True)
    pose = poses[frame_idx].to(device, non_blocking=True)
    directions = ray_dirs_cam[ys.to(device), xs.to(device)]
    rays_d = torch.bmm(pose[:, :3, :3], directions.unsqueeze(-1)).squeeze(-1)
    rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)
    result = (pose[:, :3, 3], rays_d, rgb, depth, mask)
    return (*result, {"edge_pixel_fraction": actual_edge_fraction}) if return_sampling_stats else result


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


@torch.inference_mode()
def evaluate_model(
    model: HierarchicalNeRF,
    images: torch.Tensor,
    depths: torch.Tensor,
    masks: torch.Tensor,
    poses: torch.Tensor,
    ray_dirs_cam: torch.Tensor,
    config: ExperimentConfig,
    frame_indices: list[int],
) -> tuple[list[dict], list[tuple], float, float]:
    metrics = []
    rendered = []
    model.eval()
    for idx in frame_indices:
        device = ray_dirs_cam.device
        pred_rgb, pred_depth = render_image(model, poses[idx].to(device), ray_dirs_cam, config)
        gt_rgb, gt_depth, gt_mask = images[idx].to(device), depths[idx].to(device), masks[idx].to(device) > 0.5
        psnr = -10.0 * math.log10(max(F.mse_loss(pred_rgb, gt_rgb).item(), 1e-12))
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


def train_model(model: HierarchicalNeRF, baseline_model: HierarchicalNeRF, dataset: LoadedDataset, ray_dirs_cam: torch.Tensor, config: ExperimentConfig, paths: PathsConfig, device: torch.device, external_resume: Path | None = None, scheduler_policy: str = "native", immutable_steps: set[int] | None = None, continuation: dict | None = None) -> Path:
    train_cfg = config.train
    sampler_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    edge_maps = None
    if train_cfg.edge_sampling_fraction > 0.0:
        edge_maps = build_sobel_edge_maps(dataset.train_images, train_cfg.edge_threshold)
        atomic_write_json(paths.outputs_dir / "edge_sampling.json", {
            "algorithm": "per-image luminance Sobel magnitude (L2), normalized by each image maximum",
            "luminance": [0.2126, 0.7152, 0.0722],
            "threshold": train_cfg.edge_threshold,
            "requested_edge_fraction": train_cfg.edge_sampling_fraction,
            "training_tensor_shape": list(dataset.train_images.shape),
            "edge_pixels": int(edge_maps.sum().item()),
            "total_training_pixels": int(edge_maps.numel()),
            "edge_map_fraction": float(edge_maps.float().mean().item()),
            "provenance": "official training RGB tensors only; validation and official test frames excluded",
        })
    if not train_cfg.on_demand_ray_sampling:
        train_rays_o, train_rays_d, train_rgb_pool, train_depth_pool, train_mask_pool = build_ray_pool(dataset.train_images, dataset.train_depths, dataset.train_masks, dataset.train_poses, ray_dirs_cam)
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
    val_frame_indices = list(range(min(train_cfg.val_frame_count, len(dataset.val_images))))
    current_signature = config.signature()
    best_val_psnr = float("-inf")
    best_val_step = -1
    optimizer_step_count = 0
    continuation_success_updates = attempted_iterations = skipped_amp_updates = 0
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    resumed = False
    last_preview: str | None = None
    telemetry = RunTelemetry(paths.outputs_dir / "live", audit_dir=paths.outputs_dir / "e8_logs")
    graceful_stop_request = paths.outputs_dir / "graceful_stop.request"
    started_at = time.perf_counter()
    atomic_write_json(telemetry.directory / "observed_rgbd_reference.json", scene_snapshot(
        dataset.train_images, dataset.train_depths, dataset.train_masks, dataset.train_poses, ray_dirs_cam
    ))

    source_resume = external_resume if external_resume is not None else resume_path
    if train_cfg.resume and source_resume.exists():
        resume_state = safe_torch_load(source_resume, device)
        if resume_state.get("config_signature") == current_signature or scheduler_policy == "controlled_extension_scheduler" or continuation:
            model.load_state_dict(resume_state["model_state_dict"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            if scheduler_policy == "controlled_extension_scheduler":
                schedule_state = resume_state.get("extension_schedule", {})
                initial = float(schedule_state.get("initial_lr", 4e-5))
                final = float(schedule_state.get("final_lr", 4e-6))
                # A resume checkpoint retains its original 12k extension origin;
                # do not restart the decay at a later E7 checkpoint.
                start = int(schedule_state.get("start_step", int(resume_state.get("step", -1)) + 1))
                horizon = int(schedule_state.get("end_step", train_cfg.num_steps))
                restored_lrs = [group["lr"] for group in optimizer.param_groups]
                for group in optimizer.param_groups: group["lr"] = initial
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (final + (initial-final)*.5*(1+math.cos(math.pi*min(max(epoch,0),horizon-start)/max(horizon-start,1)))) / train_cfg.lr)
                if schedule_state and resume_state.get("scheduler_state_dict"):
                    scheduler.load_state_dict(resume_state["scheduler_state_dict"])
                    for group, lr in zip(optimizer.param_groups, restored_lrs): group["lr"] = lr
                atomic_write_json(paths.outputs_dir / "extension_schedule.json", {"policy": scheduler_policy, "start_step": start, "end_step": horizon, "initial_lr": initial, "final_lr": final})
            else: scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            if resume_state.get("scaler_state_dict"):
                scaler.load_state_dict(resume_state["scaler_state_dict"])
            loss_history = list(resume_state.get("loss_history", []))
            optimizer_step_count = int(resume_state.get("optimizer_step_count", 0))
            if not optimizer_step_count:
                recorded_steps = [int(state["step"]) for state in optimizer.state_dict()["state"].values() if "step" in state]
                optimizer_step_count = max(recorded_steps, default=0)
            if resume_state.get("sampler_generator_state") is not None:
                # map_location=device moves every tensor during CUDA loads;
                # this generator is explicitly CPU-backed.
                sampler_generator.set_state(resume_state["sampler_generator_state"].cpu())
            restore_rng_state(resume_state.get("rng_state", {}))
            if resume_state.get("best_model_state_dict") is not None:
                best_state_dict = {k: v.detach().cpu().clone() for k, v in resume_state["best_model_state_dict"].items()}
            best_val_psnr = float(resume_state.get("best_val_psnr", best_val_psnr))
            best_val_step = int(resume_state.get("best_val_step", best_val_step))
            start_step = int(resume_state.get("step", -1)) + 1
            if continuation:
                continuation_success_updates = int(resume_state.get("continuation_success_updates", 0))
                attempted_iterations = int(resume_state.get("attempted_iterations", 0))
                skipped_amp_updates = int(resume_state.get("skipped_amp_updates", 0))
                start_lr, final_lr, budget = float(continuation["start_lr"]), float(continuation["final_lr"]), int(continuation["successful_updates"])
                if continuation.get("resume_existing"):
                    prior = dict(resume_state.get("continuation") or {}); prior.pop("resume_existing", None)
                    expected = dict(continuation); expected.pop("resume_existing", None)
                    if prior != expected: raise ValueError("continuation schedule provenance mismatch")
                elif scheduler.state_dict().get("last_epoch") < 48000: raise ValueError("E8 continuation requires a completed source scheduler")
                # Adam moments survive, but the source scheduler's initial_lr
                # must never become the base of this independent schedule.
                for group in optimizer.param_groups:
                    group["lr"] = start_lr
                    group["initial_lr"] = start_lr
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (final_lr + (start_lr-final_lr)*.5*(1+math.cos(math.pi*min(epoch,budget)/budget))) / start_lr)
                if continuation.get("resume_existing"):
                    scheduler.load_state_dict(resume_state["scheduler_state_dict"])
                    for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()): group["lr"] = lr
            resumed = True
            print(f"Resumed training from step {start_step} using {source_resume}")
        else:
            print("Resume checkpoint config mismatch detected. Ignoring stale resume state and starting fresh.")
    else:
        print("Starting fresh HierarchicalNeRF training run.")
    telemetry.status("running", start_step, total_steps=train_cfg.num_steps, device=str(device), resumed=resumed)

    bootstrap_metrics, rendered, bootstrap_avg_psnr, bootstrap_avg_depth_mae = evaluate_model(
        model, dataset.val_images, dataset.val_depths, dataset.val_masks, dataset.val_poses, ray_dirs_cam, config, val_frame_indices
    )
    save_eval_figure(rendered, bootstrap_metrics, bootstrap_avg_psnr, bootstrap_avg_depth_mae, val_dir / "bootstrap_eval.png")
    save_evaluation_metrics(paths, bootstrap_metrics, bootstrap_avg_psnr, bootstrap_avg_depth_mae, "bootstrap_metrics.json")
    # A bootstrap render is diagnostic only after a resume.  In particular, do
    # not replace the best state restored from the checkpoint: its validation
    # score may be better than the state at the interruption point.
    if not resumed:
        best_val_psnr = bootstrap_avg_psnr
        best_val_step = -1
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        atomic_torch_save(
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
    end_step = min(train_cfg.num_steps, train_cfg.stop_after_step) if train_cfg.stop_after_step is not None else train_cfg.num_steps
    if train_cfg.accumulation_steps < 1: raise ValueError("accumulation_steps must be positive")
    iteration = start_step
    while (continuation_success_updates < int(continuation["successful_updates"]) if continuation else iteration < end_step):
        step = iteration; iteration += 1; attempted_iterations += 1
        sampling_stats: dict[str, float] = {}; optimizer.zero_grad(set_to_none=True)
        totals = {key: 0.0 for key in ("fine_rgb_loss","coarse_rgb_loss","fine_depth_loss","coarse_depth_loss","loss")}
        for micro_step in range(train_cfg.accumulation_steps):
            if train_cfg.on_demand_ray_sampling:
                batch = sample_ray_batch(dataset.train_images, dataset.train_depths, dataset.train_masks, dataset.train_poses, ray_dirs_cam, train_cfg.batch_rays, device, sampler_generator, edge_maps, train_cfg.edge_sampling_fraction, return_sampling_stats=edge_maps is not None)
                rays_o, rays_d, rgb_gt, depth_gt, train_mask = batch[:5]; sampling_stats = batch[5] if edge_maps is not None else {}
            else:
                idx = torch.randint(0, train_rays_o.shape[0], (train_cfg.batch_rays,), device=device); rays_o, rays_d, rgb_gt, depth_gt, train_mask = train_rays_o[idx], train_rays_d[idx], train_rgb_pool[idx], train_depth_pool[idx], train_mask_pool[idx]
            depth_mask = train_mask > 0.5
            with autocast(device_type=device.type, enabled=(train_cfg.use_amp and device.type == "cuda")):
                rgb_pred, depth_pred, _, coarse_rgb, coarse_depth, _ = render_rays(model, rays_o, rays_d, config, perturb=True, return_coarse=True)
                fine_rgb_loss = F.mse_loss(rgb_pred, rgb_gt); coarse_rgb_loss = F.mse_loss(coarse_rgb, rgb_gt)
                fine_depth_loss = F.smooth_l1_loss(depth_pred[depth_mask], depth_gt[depth_mask]) if depth_mask.any() else torch.zeros((), device=device)
                coarse_depth_loss = F.smooth_l1_loss(coarse_depth[depth_mask], depth_gt[depth_mask]) if depth_mask.any() else torch.zeros((), device=device)
                loss = train_cfg.rgb_loss_weight * (fine_rgb_loss + coarse_rgb_loss) + train_cfg.depth_loss_weight * (fine_depth_loss + coarse_depth_loss)
            scaler.scale(loss / train_cfg.accumulation_steps).backward()
            for key, value in (("fine_rgb_loss",fine_rgb_loss),("coarse_rgb_loss",coarse_rgb_loss),("fine_depth_loss",fine_depth_loss),("coarse_depth_loss",coarse_depth_loss),("loss",loss)): totals[key] += float(value.item())
        fine_rgb_loss, coarse_rgb_loss, fine_depth_loss, coarse_depth_loss = (torch.tensor(totals[key] / train_cfg.accumulation_steps, device=device) for key in ("fine_rgb_loss","coarse_rgb_loss","fine_depth_loss","coarse_depth_loss"))
        loss = torch.tensor(totals["loss"] / train_cfg.accumulation_steps, device=device)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # GradScaler skips optimizer.step on overflow.  Advancing the scheduler
        # in that case desynchronizes LR state from actual parameter updates.
        # A scale backoff is the documented observable for a skipped step.
        optimizer_stepped = (not scaler.is_enabled()) or scaler.get_scale() >= scale_before
        if optimizer_stepped:
            optimizer_step_count += 1
            if continuation: continuation_success_updates += 1
            scheduler.step()
        else:
            skipped_amp_updates += 1
        loss_history.append(loss.item())
        # This wall-clock-only pause happens after all numerical state for the
        # update is committed. It deliberately cannot change gradients,
        # sampling draws, scheduler steps, or the serialized checkpoint.
        if train_cfg.optimizer_pause_seconds > 0:
            time.sleep(train_cfg.optimizer_pause_seconds)
        if (step + 1) % min(25, train_cfg.checkpoint_every) == 0 or step + 1 == end_step:
            allocated = reserved = 0.0
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device) / 2**20
                reserved = torch.cuda.memory_reserved(device) / 2**20
            telemetry.publish(
                "train_step", step + 1, total_steps=train_cfg.num_steps,
                coarse_rgb_loss=float(coarse_rgb_loss.item()), fine_rgb_loss=float(fine_rgb_loss.item()),
                coarse_depth_loss=float(coarse_depth_loss.item()), fine_depth_loss=float(fine_depth_loss.item()),
                loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]),
                cuda_allocated_mb=allocated, cuda_reserved_mb=reserved,
                optimizer_step_count=optimizer_step_count, continuation_success_updates=continuation_success_updates, attempted_iterations=attempted_iterations, skipped_amp_updates=skipped_amp_updates, amp_step_skipped=not optimizer_stepped,
                **sampling_stats,
                elapsed_seconds=time.perf_counter() - started_at,
            )

        stop_requested = graceful_stop_request.exists()
        should_checkpoint = ((step + 1) % train_cfg.checkpoint_every == 0) or ((step + 1) == end_step) or stop_requested
        should_validate = ((step + 1) % train_cfg.val_render_interval == 0) or ((step + 1) == end_step)
        if should_validate:
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
            metrics, rendered, avg_psnr, avg_depth_mae = evaluate_model(
                model, dataset.val_images, dataset.val_depths, dataset.val_masks, dataset.val_poses, ray_dirs_cam, config, val_frame_indices
            )
            save_eval_figure(rendered, metrics, avg_psnr, avg_depth_mae, val_dir / f"hierarchical_nerf_val_{step + 1:05d}.png")
            save_json(
                val_dir / f"hierarchical_nerf_val_{step + 1:05d}.json",
                {"frames": metrics, "average_psnr": avg_psnr, "average_depth_mae": avg_depth_mae},
            )
            predicted_scene_name = f"nerf_predicted_step_{step + 1:05d}.json"
            atomic_write_json(
                telemetry.directory / predicted_scene_name,
                predicted_scene_snapshot(rendered, dataset.val_poses, ray_dirs_cam, step + 1, config.scene.near, config.scene.far),
            )
            telemetry.publish(
                "validation", step + 1, total_steps=train_cfg.num_steps, validation_psnr=avg_psnr,
                validation_depth_mae=avg_depth_mae,
                cuda_peak_allocated_mb=(torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0),
                cuda_peak_reserved_mb=(torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0),
                preview=f"visualizations/train_validation/hierarchical_nerf_val_{step + 1:05d}.png",
                observed_scene="live/observed_rgbd_reference.json", nerf_scene=f"live/{predicted_scene_name}",
                depth_registration_note="Raw 7-Scenes RGB/depth cameras are not perfectly registered.",
            )
            last_preview = f"visualizations/train_validation/hierarchical_nerf_val_{step + 1:05d}.png"
            if avg_psnr > best_val_psnr:
                best_val_psnr = avg_psnr
                best_val_step = step
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                atomic_torch_save(
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
            _require_checkpoint_disk_space(resume_path)
            state = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "best_model_state_dict": best_state_dict,
                    "best_val_psnr": best_val_psnr,
                    "best_val_step": best_val_step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "optimizer_step_count": optimizer_step_count,
                    "continuation_success_updates": continuation_success_updates, "attempted_iterations": attempted_iterations, "skipped_amp_updates": skipped_amp_updates, "continuation": continuation,
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss_history": loss_history,
                    "sampler_generator_state": sampler_generator.get_state(),
                    "rng_state": capture_rng_state(),
                    "config": config.to_dict(),
                    "config_signature": current_signature, "scheduler_policy": scheduler_policy,
                    "extension_schedule": ({"start_step": start, "end_step": horizon, "initial_lr": initial, "final_lr": final} if scheduler_policy == "controlled_extension_scheduler" else None),
                    "parent_checkpoint": str(external_resume) if external_resume else None,
                }
            atomic_torch_save(state, resume_path)
            if immutable_steps and step + 1 in immutable_steps:
                snap = paths.checkpoints_dir / f"hierarchical_nerf_step_{step + 1:05d}.pth"
                if snap.exists(): raise FileExistsError(f"immutable snapshot exists: {snap}")
                atomic_torch_save(state, snap)
                _record_immutable_snapshot(paths.checkpoints_dir, snap)
        if stop_requested:
            telemetry.status("stopped", step + 1, total_steps=train_cfg.num_steps, checkpoint=str(resume_path), reason="graceful_stop.request", preview=last_preview, observed_scene="live/observed_rgbd_reference.json", nerf_scene=f"live/nerf_predicted_step_{step + 1:05d}.json")
            return resume_path

    if end_step < train_cfg.num_steps:
        telemetry.status("stopped", end_step, total_steps=train_cfg.num_steps, checkpoint=str(resume_path), preview=last_preview, observed_scene="live/observed_rgbd_reference.json", nerf_scene=f"live/nerf_predicted_step_{end_step:05d}.json")
        return resume_path
    if not continuation:
        model.load_state_dict(best_state_dict)
    if continuation:
        checkpoint_path = paths.checkpoints_dir / "hierarchical_nerf_continuation_trained.pth"
    atomic_torch_save(
        {
            "step": best_val_step if best_val_step >= 0 else train_cfg.num_steps - 1,
            "best_val_psnr": best_val_psnr,
            "best_val_step": best_val_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_step_count": optimizer_step_count,
            "continuation_success_updates": continuation_success_updates, "attempted_iterations": attempted_iterations, "skipped_amp_updates": skipped_amp_updates, "continuation": continuation,
            "export_kind": "terminal_continuation_state" if continuation else "best_validation_state",
            "source_sha256": continuation.get("source_sha256") if continuation else None,
            "depth_loss_weight": config.train.depth_loss_weight,
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss_history": loss_history,
            "sampler_generator_state": sampler_generator.get_state(),
            "rng_state": capture_rng_state(),
            "config": config.to_dict(),
            "config_signature": current_signature,
        },
        checkpoint_path,
    )
    baseline_model.load_state_dict(model.state_dict())
    baseline_model.eval()
    telemetry.status("completed", train_cfg.num_steps, total_steps=train_cfg.num_steps, checkpoint=str(checkpoint_path), best_validation_psnr=best_val_psnr, best_validation_step=best_val_step, elapsed_seconds=time.perf_counter() - started_at, preview=last_preview, observed_scene="live/observed_rgbd_reference.json", nerf_scene=f"live/nerf_predicted_step_{end_step:05d}.json")
    print(f"Saved trained checkpoint to: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model: HierarchicalNeRF, baseline_model: HierarchicalNeRF, checkpoint_path: Path, device: torch.device, config: ExperimentConfig) -> dict:
    checkpoint = safe_torch_load(checkpoint_path, device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a model_state_dict.")
    if checkpoint.get("config_signature") != config.signature():
        raise ValueError(f"Checkpoint {checkpoint_path} is incompatible with the active experiment configuration.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    baseline_model.load_state_dict(model.state_dict())
    baseline_model.eval()
    model.eval()
    return checkpoint
