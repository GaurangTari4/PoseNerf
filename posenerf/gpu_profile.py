"""Bounded, non-destructive CUDA training-step profiler for PoseNeRF."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .model import HierarchicalNeRF, render_rays
from .utils import save_json, seed_everything


def _smi() -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _config(width: int, layers: int, rays: int) -> ExperimentConfig:
    config = ExperimentConfig(seed=2027)
    config.scene.downsample = 4
    config.train.batch_rays = rays
    config.train.max_train_images = 192
    config.train.max_val_images = 12
    config.train.num_steps = 12000
    config.train.num_coarse_samples = config.train.num_fine_samples = 48
    config.train.depth_loss_weight = 0.03
    config.train.use_amp = True
    config.model.hidden, config.model.layers = width, layers
    return config


def profile_trial(width: int, layers: int, rays: int, warmup: int, measured_steps: int, budget_bytes: int) -> dict:
    """Profile one full training update path, never saving model/checkpoint state."""
    device = torch.device("cuda")
    seed_everything(2027)
    torch.cuda.empty_cache()  # profiler isolation only; not a training recommendation
    free_before, total = torch.cuda.mem_get_info(device)
    config = _config(width, layers, rays)
    model = HierarchicalNeRF(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    rays_o = torch.zeros((rays, 3), device=device)
    rays_d = F.normalize(torch.randn((rays, 3), device=device), dim=-1)
    rgb_target = torch.rand((rays, 3), device=device)
    depth_target = torch.rand((rays,), device=device) * 4.4 + 0.1
    last_loss = [float("nan")]
    torch.cuda.reset_peak_memory_stats(device)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            rgb, depth, _, coarse_rgb, coarse_depth, _ = render_rays(model, rays_o, rays_d, config, perturb=True, return_coarse=True)
            loss = F.mse_loss(rgb, rgb_target) + F.mse_loss(coarse_rgb, rgb_target)
            loss = loss + config.train.depth_loss_weight * (F.smooth_l1_loss(depth, depth_target) + F.smooth_l1_loss(coarse_depth, depth_target))
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        if not torch.isfinite(loss): raise RuntimeError("non-finite training loss")
        last_loss[0] = float(loss.detach())

    try:
        for _ in range(warmup): step()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(measured_steps): step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device); reserved = torch.cuda.memory_reserved(device)
        return {
            "width": width, "layers": layers, "batch_rays": rays, "status": "ok",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "steps": measured_steps, "elapsed_seconds": elapsed, "steps_per_second": measured_steps / elapsed,
            "allocated_bytes": allocated, "reserved_bytes": reserved,
            "peak_allocated_bytes": peak_alloc, "peak_reserved_bytes": peak_reserved,
            "free_before_bytes": free_before, "dedicated_total_bytes": total,
            "budget_bytes": budget_bytes, "within_budget": peak_reserved <= budget_bytes,
            "loss": last_loss[0],
        }
    except torch.OutOfMemoryError as error:
        return {"width": width, "layers": layers, "batch_rays": rays, "status": "oom", "message": str(error), "free_before_bytes": free_before, "dedicated_total_bytes": total, "budget_bytes": budget_bytes}
    finally:
        del model, optimizer, rays_o, rays_d, rgb_target, depth_target
        torch.cuda.empty_cache()  # release only after the isolated trial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--include-8192", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for GPU profiling")
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device("cuda"); properties = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    # The smaller of 80% dedicated capacity and 80% currently-free capacity
    # protects Studio/other processes and never treats shared Windows memory as VRAM.
    budget = min(int(total * 0.80), int(free * 0.80))
    plan = [(192, 8, 1024), (256, 8, 1024), (384, 8, 1024), (192, 8, 2048), (192, 8, 4096)]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": {"name": properties.name, "dedicated_total_bytes": properties.total_memory, "compute_capability": [properties.major, properties.minor], "multiprocessors": properties.multi_processor_count}, "initial_mem_get_info": {"free_bytes": free, "total_bytes": total}, "budget_bytes": budget, "budget_policy": "min(80% dedicated VRAM, 80% currently free dedicated VRAM)", "nvidia_smi_before": _smi(), "trials": []}
    for width, layers, rays in plan:
        result = profile_trial(width, layers, rays, args.warmup, args.steps, budget)
        report["trials"].append(result)
    baseline_4096 = next((x for x in report["trials"] if x["width"] == 192 and x["batch_rays"] == 4096), None)
    if args.include_8192 and baseline_4096 and baseline_4096["status"] == "ok" and baseline_4096["peak_reserved_bytes"] < budget * 0.50:
        report["trials"].append(profile_trial(192, 8, 8192, args.warmup, args.steps, budget))
    else:
        report["8192_decision"] = "not attempted: requires an explicitly safe 4096-ray result below 50% of the conservative budget"
    report["nvidia_smi_after"] = _smi()
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
