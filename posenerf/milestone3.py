"""Reproducible Milestone 3 evaluation and single-factor pilot experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig, build_paths, ensure_directories
from .data import load_chess_dataset
from .model import HierarchicalNeRF, get_ray_directions, render_image
from .stage_c import stage_paths
from .training import train_model
from .utils import get_device, safe_torch_load, save_json, seed_everything


BASELINE_RUN = "stage_d_b1_coverage_192_a3_extended_12k"


def _safe_run(workspace: Path, name: str) -> Path:
    """Resolve an existing named run without accepting paths outside outputs."""
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("run name must be a single outputs-directory name")
    run = (workspace / "outputs" / name).resolve()
    outputs = (workspace / "outputs").resolve()
    if outputs not in run.parents or not run.is_dir() or not (run / "manifest.json").is_file():
        raise ValueError(f"completed run not found: {name}")
    return run


def _new_output(workspace: Path, output: Path) -> Path:
    output = output.resolve()
    outputs = (workspace / "outputs").resolve()
    if outputs not in output.parents or output.exists():
        raise ValueError("output must be a new directory below workspace/outputs")
    return output


def _checkpoint_for(run: Path, name: str, config: ExperimentConfig, device: torch.device) -> tuple[Path, dict]:
    if Path(name).name != name or not name.endswith(".pth"):
        raise ValueError("checkpoint must be a checkpoint filename, not a path")
    checkpoint = run / "checkpoints" / name
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {name}")
    payload = safe_torch_load(checkpoint, device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("checkpoint does not contain model weights")
    recorded = payload.get("config_signature")
    expected = config.signature()
    # Some completed runs predate configuration fields that were later added
    # with semantics-preserving defaults.  Accept only those narrowly scoped
    # legacy signatures so a read-only evaluator remains able to assess the
    # immutable historical checkpoints.  Any recorded non-default field still
    # has to match the run manifest exactly.
    if isinstance(recorded, dict):
        for field in ("edge_sampling_fraction", "edge_threshold", "accumulation_steps", "optimizer_pause_seconds"):
            if field not in recorded:
                expected.pop(field, None)
    if recorded != expected:
        raise ValueError("checkpoint configuration signature does not match its run manifest")
    return checkpoint, payload


def _apply(value, source: dict) -> None:
    for item in fields(value):
        if item.name not in source:
            continue
        current = getattr(value, item.name)
        incoming = source[item.name]
        if hasattr(current, "__dataclass_fields__") and isinstance(incoming, dict):
            _apply(current, incoming)
        else:
            setattr(value, item.name, incoming)


def load_run_config(run: Path) -> ExperimentConfig:
    config = ExperimentConfig()
    _apply(config, json.loads((run / "manifest.json").read_text(encoding="utf-8"))["config"])
    return config


def _ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Image-global SSIM, dependency-free and explicitly reported as such."""
    x, y = pred.float(), target.float()
    mu_x, mu_y = x.mean((0, 1)), y.mean((0, 1))
    var_x, var_y = x.var((0, 1), correction=0), y.var((0, 1), correction=0)
    cov = ((x - mu_x) * (y - mu_y)).mean((0, 1))
    c1, c2 = 0.01**2, 0.03**2
    return float((((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2))).mean())


def _save_comparison(path: Path, samples: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
    rows = len(samples)
    figure, axes = plt.subplots(rows, 5, figsize=(18, 3.5 * rows), squeeze=False)
    titles = ["Reference RGB", "Predicted RGB", "Absolute RGB Error", "Reference Depth", "Predicted Depth"]
    for row, (name, rgb, pred, depth, pred_depth) in enumerate(samples):
        error = (rgb - pred).abs()
        values = [rgb, pred, error / max(float(error.max()), 1e-6), depth, pred_depth]
        for col, value in enumerate(values):
            ax = axes[row, col]
            ax.imshow(value.cpu().numpy(), cmap="magma" if col >= 3 else None)
            ax.set_title(titles[col] if row == 0 else "")
            ax.set_ylabel(name if col == 0 else "")
            ax.axis("off")
    figure.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=160); plt.close(figure)


@torch.inference_mode()
def evaluate_frozen(
    run: Path,
    output: Path,
    checkpoint_name: str = "hierarchical_nerf_best.pth",
    evaluation_downsample: int | None = None,
    frame_indices: tuple[int, ...] | None = None,
) -> dict:
    """Measure a checkpoint against the frozen frames at one explicit resolution.

    ``evaluation_downsample`` changes only the evaluation image/ray grid.  The
    checkpoint architecture is unchanged, so this makes it possible to compare
    a lower-resolution-trained model fairly with a higher-resolution model at
    either common target resolution.
    """
    config = load_run_config(run)
    trained_downsample = config.scene.downsample
    if evaluation_downsample is not None:
        config.scene.downsample = evaluation_downsample
    workspace = run.parent.parent
    paths = build_paths(workspace)
    device = get_device(); seed_everything(config.seed)
    dataset = load_chess_dataset(config, paths, device, storage_device=torch.device("cpu"), include_test=False)
    frozen = json.loads((run / "frozen_protocol.json").read_text(encoding="utf-8"))
    actual_ids = [str(frame.color_path.relative_to(paths.scene_dir)).replace("\\", "/") for frame in dataset.val_frames]
    expected_ids = [name.replace("\\", "/") for name in frozen["validation_frames"]]
    if actual_ids != expected_ids:
        raise ValueError("loaded validation frames do not match the frozen protocol")
    checkpoint, payload = _checkpoint_for(run, checkpoint_name, config, device)
    model = HierarchicalNeRF(config).to(device); model.load_state_dict(payload["model_state_dict"], strict=True); model.eval()
    directions = get_ray_directions(config, device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    indices = tuple(range(len(actual_ids))) if frame_indices is None else tuple(frame_indices)
    if not indices or any(index < 0 or index >= len(actual_ids) for index in indices):
        raise ValueError("frame_indices must be non-empty frozen-frame indices")
    rows, visuals, latencies = [], [], []
    for position, index in enumerate(indices):
        frame_id = actual_ids[index]
        started = time.perf_counter()
        pred, pred_depth = render_image(model, dataset.val_poses[index].to(device), directions, config)
        if device.type == "cuda": torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - started) * 1000)
        rgb, depth, mask = dataset.val_images[index].to(device), dataset.val_depths[index].to(device), dataset.val_masks[index].to(device) > 0.5
        mse = float(F.mse_loss(pred, rgb)); depth_mae = float((pred_depth[mask] - depth[mask]).abs().mean()) if mask.any() else float("nan")
        rows.append({"frame": frame_id, "psnr": -10 * math.log10(max(mse, 1e-12)), "ssim_global": _ssim(pred, rgb), "rgb_mse": mse, "depth_mae": depth_mae, "latency_ms": latencies[-1]})
        if position in (0, len(indices) // 2, len(indices) - 1): visuals.append((frame_id, rgb, pred, depth, pred_depth))
    _save_comparison(output / "visualizations" / "frozen_validation_comparison.png", visuals)
    valid_depth = [row["depth_mae"] for row in rows if math.isfinite(row["depth_mae"])]
    report = {
        "kind": "newly_measured_frozen_validation_baseline", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": run.name, "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "checkpoint_step": int(payload.get("step", -1)),
        "trained_downsample": trained_downsample, "evaluation_downsample": config.scene.downsample,
        "evaluation_resolution": [config.scene.width, config.scene.height],
        "frozen_protocol_sha256": frozen.get("protocol_sha256"), "validation_frames": [actual_ids[index] for index in indices],
        "metrics": {"psnr": float(np.mean([x["psnr"] for x in rows])), "ssim_global": float(np.mean([x["ssim_global"] for x in rows])), "lpips": None, "depth_mae": float(np.mean(valid_depth)), "rgb_mse": float(np.mean([x["rgb_mse"] for x in rows])), "render_latency_ms_mean": float(np.mean(latencies)), "render_latency_ms_p95": float(np.percentile(latencies, 95)), "cuda_peak_mb": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0},
        "notes": ["SSIM is dependency-free image-global SSIM.", "LPIPS unavailable: dependency not installed.", "All metrics above were measured in this invocation, not copied from historical telemetry.", "Evaluation rays and resized reference images use the same scaled intrinsics."],
        "frames": rows,
    }
    save_json(output / "baseline_report.json", report)
    return report


def run_coverage_pilot(workspace: Path, name: str, steps: int, train_images: int, resume: bool = False) -> Path:
    output_paths = stage_paths(workspace, name)
    baseline = workspace / "outputs" / BASELINE_RUN
    if output_paths.outputs_dir.exists() and not resume: raise FileExistsError(f"refusing to overwrite {output_paths.outputs_dir}")
    ensure_directories(output_paths)
    config = load_run_config(output_paths.outputs_dir) if resume else load_run_config(baseline)
    config.train.max_train_images = train_images; config.train.num_steps = steps; config.train.stop_after_step = steps
    # Retain B1's optimizer horizon, checkpoint cadence, and two-frame
    # training-time selection metric.  The separate frozen evaluator measures
    # all twelve validation frames after training.
    config.train.resume = resume
    manifest = output_paths.outputs_dir / "manifest.json"
    if not manifest.exists(): save_json(manifest, {"stage": "M3", "kind": "coverage_only_fresh_init_pilot", "created_utc": datetime.now(timezone.utc).isoformat(), "base_run": BASELINE_RUN, "config": config.to_dict(), "platform": platform.platform(), "torch": torch.__version__})
    device = get_device(); seed_everything(config.seed)
    dataset = load_chess_dataset(config, output_paths, device, storage_device=torch.device("cpu"), include_test=False)
    baseline_protocol = json.loads((baseline / "frozen_protocol.json").read_text(encoding="utf-8"))
    ids = [str(frame.color_path.relative_to(output_paths.scene_dir)).replace("\\", "/") for frame in dataset.val_frames]
    if ids != [x.replace("\\", "/") for x in baseline_protocol["validation_frames"]]: raise ValueError("pilot validation protocol drift")
    protocol_path = output_paths.outputs_dir / "frozen_protocol.json"
    if not protocol_path.exists(): save_json(protocol_path, {**baseline_protocol, "validation_frames": ids, "pilot_train_frames": [str(x.color_path.relative_to(output_paths.scene_dir)).replace("\\", "/") for x in dataset.train_frames], "fresh_initialization": True})
    directions = get_ray_directions(config, device); model = HierarchicalNeRF(config).to(device); reference = HierarchicalNeRF(config).to(device)
    train_model(model, reference, dataset, directions, config, output_paths, device)
    return output_paths.outputs_dir


def compare_reports(baseline: Path, candidate: Path, output: Path) -> None:
    left = json.loads((baseline / "baseline_report.json").read_text(encoding="utf-8"))
    right = json.loads((candidate / "baseline_report.json").read_text(encoding="utf-8"))
    keys = ("psnr", "ssim_global", "depth_mae", "rgb_mse", "render_latency_ms_mean", "cuda_peak_mb")
    save_json(output, {"shared_frozen_protocol": left["frozen_protocol_sha256"] == right["frozen_protocol_sha256"], "baseline": left["source_run"], "candidate": right["source_run"], "metrics": {key: {"baseline": left["metrics"][key], "candidate": right["metrics"][key], "delta_candidate_minus_baseline": right["metrics"][key] - left["metrics"][key]} for key in keys}, "interpretation": "Candidate is a 300-step fresh-init pilot; it is not duration-matched to the 12k-step baseline and cannot support a coverage-quality conclusion."})


@torch.inference_mode()
def compare_completed_runs(workspace: Path, baseline_report_dir: Path, candidate_report_dir: Path, output: Path, edge_threshold: float = 0.20) -> None:
    """Compare arbitrary newly measured completed runs under one edge mask."""
    baseline = json.loads((baseline_report_dir / "baseline_report.json").read_text(encoding="utf-8"))
    candidate = json.loads((candidate_report_dir / "baseline_report.json").read_text(encoding="utf-8"))
    if baseline["validation_frames"] != candidate["validation_frames"] or baseline["evaluation_resolution"] != candidate["evaluation_resolution"]:
        raise ValueError("reports do not share frozen frames and resolution")
    if baseline["frozen_protocol_sha256"] != candidate["frozen_protocol_sha256"]:
        raise ValueError("reports do not share a frozen protocol")
    if not 0.0 <= edge_threshold <= 1.0:
        raise ValueError("edge threshold must be in [0, 1]")
    base_run = _safe_run(workspace, baseline["source_run"]); candidate_run = _safe_run(workspace, candidate["source_run"])
    base_config, candidate_config = load_run_config(base_run), load_run_config(candidate_run)
    if base_config.scene.downsample != candidate_config.scene.downsample:
        raise ValueError("models were evaluated on incompatible image grids")
    device = get_device(); paths = build_paths(workspace)
    dataset = load_chess_dataset(base_config, paths, device, storage_device=torch.device("cpu"), include_test=False)
    base_checkpoint, base_payload = _checkpoint_for(base_run, Path(baseline["checkpoint"]).name, base_config, device)
    candidate_checkpoint, candidate_payload = _checkpoint_for(candidate_run, Path(candidate["checkpoint"]).name, candidate_config, device)
    base_model = HierarchicalNeRF(base_config).to(device); base_model.load_state_dict(base_payload["model_state_dict"], strict=True); base_model.eval()
    candidate_model = HierarchicalNeRF(candidate_config).to(device); candidate_model.load_state_dict(candidate_payload["model_state_dict"], strict=True); candidate_model.eval()
    base_dirs, candidate_dirs = get_ray_directions(base_config, device), get_ray_directions(candidate_config, device)
    rows, plot_rows = [], []
    for index, frame in enumerate(baseline["validation_frames"]):
        rgb, depth, mask = dataset.val_images[index].to(device), dataset.val_depths[index].to(device), dataset.val_masks[index].to(device) > 0.5
        base_rgb, base_depth = render_image(base_model, dataset.val_poses[index].to(device), base_dirs, base_config)
        candidate_rgb, candidate_depth = render_image(candidate_model, dataset.val_poses[index].to(device), candidate_dirs, candidate_config)
        gray = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
        gx = F.pad(gray[:, 2:] - gray[:, :-2], (1, 1)); gy = F.pad(gray[2:, :] - gray[:-2, :], (0, 0, 1, 1))
        magnitude = torch.hypot(gx, gy); edge = magnitude.ge(magnitude.max().clamp_min(torch.finfo(magnitude.dtype).eps) * edge_threshold)
        def values(pred: torch.Tensor, pred_depth: torch.Tensor) -> dict:
            err = (pred - rgb).abs().mean(-1)
            edge_mse = F.mse_loss(pred[edge], rgb[edge]) if edge.any() else torch.tensor(float("nan"), device=device)
            return {"psnr": -10 * math.log10(max(float(F.mse_loss(pred, rgb)), 1e-12)), "ssim_global": _ssim(pred, rgb), "rgb_mse": float(F.mse_loss(pred, rgb)), "depth_mae": float((pred_depth[mask] - depth[mask]).abs().mean()) if mask.any() else float("nan"), "edge_rgb_mae": float(err[edge].mean()) if edge.any() else float("nan"), "non_edge_rgb_mae": float(err[~edge].mean()) if (~edge).any() else float("nan"), "edge_psnr": -10 * math.log10(max(float(edge_mse), 1e-12))}
        left, right = values(base_rgb, base_depth), values(candidate_rgb, candidate_depth)
        rows.append({"frame": frame, "edge_pixel_fraction": float(edge.float().mean()), "baseline": left, "candidate": right, "delta_candidate_minus_baseline": {key: right[key] - left[key] for key in left}})
        plot_rows.append((frame, rgb, base_rgb, candidate_rgb, (rgb - base_rgb).abs(), (rgb - candidate_rgb).abs(), base_depth, candidate_depth))
    figure, axes = plt.subplots(len(plot_rows), 8, figsize=(24, 2.8 * len(plot_rows)), squeeze=False)
    titles = ("Reference RGB", "Baseline RGB", "Candidate RGB", "Baseline |error|", "Candidate |error|", "Reference depth", "Baseline depth", "Candidate depth")
    for row, values in enumerate(plot_rows):
        frame, rgb, base_rgb, candidate_rgb, base_error, candidate_error, base_depth, candidate_depth = values
        panels = (rgb, base_rgb, candidate_rgb, (base_error / 0.25).clamp(0, 1), (candidate_error / 0.25).clamp(0, 1), dataset.val_depths[row].to(device), base_depth, candidate_depth)
        for col, panel in enumerate(panels):
            ax = axes[row, col]; ax.imshow(panel.cpu().numpy(), cmap="magma" if col >= 5 else None, vmin=0.1 if col >= 5 else None, vmax=4.5 if col >= 5 else None); ax.set_title(titles[col] if row == 0 else ""); ax.set_ylabel(frame if col == 0 else ""); ax.axis("off")
    output.mkdir(parents=True, exist_ok=False); figure.tight_layout(); figure.savefig(output / "e3_vs_e5_all_frozen_views.png", dpi=160); plt.close(figure)
    keys = tuple(rows[0]["baseline"])
    summary = {key: {"baseline": float(np.mean([r["baseline"][key] for r in rows])), "candidate": float(np.mean([r["candidate"][key] for r in rows])), "delta_candidate_minus_baseline": float(np.mean([r["delta_candidate_minus_baseline"][key] for r in rows]))} for key in keys}
    frame_deltas = {key: [r["delta_candidate_minus_baseline"][key] for r in rows] for key in keys}
    save_json(output / "pairwise_report.json", {"baseline": {"run": base_run.name, "checkpoint": str(base_checkpoint)}, "candidate": {"run": candidate_run.name, "checkpoint": str(candidate_checkpoint)}, "edge_mask": {"algorithm": "reference-RGB central-difference magnitude, per-frame max normalized", "threshold": edge_threshold, "evaluation_only": True, "edge_pixel_fraction": float(np.mean([r["edge_pixel_fraction"] for r in rows]))}, "summary": summary, "frame_delta_stats": {key: {"mean": float(np.mean(values)), "median": float(np.median(values)), "improved": int(sum(value > 0 for value in values)) if key in ("psnr", "ssim_global", "edge_psnr") else int(sum(value < 0 for value in values)), "regressed": int(sum(value < 0 for value in values)) if key in ("psnr", "ssim_global", "edge_psnr") else int(sum(value > 0 for value in values))} for key, values in frame_deltas.items()}, "frames": rows})


@torch.inference_mode()
def paired_visuals(workspace: Path, candidate_name: str, evaluation_downsample: int | None = None, output_name: str = "m3_comparison", baseline_report_dir: str = "m3_b1_frozen_baseline", candidate_report_dir: str = "m3_evaluation_best") -> None:
    baseline_run, candidate_run = workspace / "outputs" / BASELINE_RUN, workspace / "outputs" / candidate_name
    base_report = json.loads((workspace / "outputs" / baseline_report_dir / "baseline_report.json").read_text(encoding="utf-8"))
    candidate_report = json.loads((candidate_run / candidate_report_dir / "baseline_report.json").read_text(encoding="utf-8"))
    if base_report["validation_frames"] != candidate_report["validation_frames"]: raise ValueError("cannot compare different validation protocols")
    base_config, candidate_config = load_run_config(baseline_run), load_run_config(candidate_run)
    if evaluation_downsample is not None:
        base_config.scene.downsample = evaluation_downsample
        candidate_config.scene.downsample = evaluation_downsample
    paths = build_paths(workspace); device = get_device()
    dataset = load_chess_dataset(base_config, paths, device, storage_device=torch.device("cpu"), include_test=False)
    def model_for(run: Path, config: ExperimentConfig) -> HierarchicalNeRF:
        payload = safe_torch_load(run / "checkpoints" / "hierarchical_nerf_best.pth", device)
        model = HierarchicalNeRF(config).to(device); model.load_state_dict(payload["model_state_dict"], strict=True); return model.eval()
    base_model, candidate_model = model_for(baseline_run, base_config), model_for(candidate_run, candidate_config)
    base_dirs, candidate_dirs = get_ray_directions(base_config, device), get_ray_directions(candidate_config, device)
    ids = (0, len(dataset.val_images) // 2, len(dataset.val_images) - 1); fig, axes = plt.subplots(len(ids), 7, figsize=(24, 3.5 * len(ids)), squeeze=False)
    for row, index in enumerate(ids):
        rgb, depth = dataset.val_images[index].to(device), dataset.val_depths[index].to(device)
        b_rgb, b_depth = render_image(base_model, dataset.val_poses[index].to(device), base_dirs, base_config)
        e_rgb, e_depth = render_image(candidate_model, dataset.val_poses[index].to(device), candidate_dirs, candidate_config)
        b_err, e_err = (rgb - b_rgb).abs(), (rgb - e_rgb).abs()
        values = [rgb, b_rgb, e_rgb, b_err / max(float(b_err.max()), 1e-6), e_err / max(float(e_err.max()), 1e-6), b_depth, e_depth]
        titles = ("Reference RGB", "B1 RGB", "Candidate RGB", "B1 |error|", "Candidate |error|", "B1 depth", "Candidate depth")
        for column, value in enumerate(values):
            axis = axes[row, column]; axis.imshow(value.cpu().numpy(), cmap="magma" if column >= 5 else None); axis.set_title(titles[column] if row == 0 else ""); axis.set_ylabel(base_report["validation_frames"][index] if column == 0 else ""); axis.axis("off")
    out = candidate_run / output_name; out.mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(out / "b1_vs_candidate_frozen_views.png", dpi=160); plt.close(fig)
    frame_delta = [{"frame": b["frame"], "psnr_delta": e["psnr"] - b["psnr"], "ssim_delta": e["ssim_global"] - b["ssim_global"], "depth_mae_delta": e["depth_mae"] - b["depth_mae"], "rgb_mse_delta": e["rgb_mse"] - b["rgb_mse"]} for b, e in zip(base_report["frames"], candidate_report["frames"])]
    save_json(out / "per_frame_deltas.json", frame_delta)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("baseline", "coverage-pilot", "compare", "pairwise", "evaluate", "pairwise-runs")); parser.add_argument("--workspace", type=Path, default=Path("workspace")); parser.add_argument("--output", type=Path, default=Path("workspace/outputs/m3_b1_frozen_baseline")); parser.add_argument("--steps", type=int, default=300); parser.add_argument("--train-images", type=int, default=256); parser.add_argument("--run-name", default="m3_e2_coverage_256_pilot_300"); parser.add_argument("--checkpoint", default="hierarchical_nerf_best.pth"); parser.add_argument("--baseline-report", type=Path); parser.add_argument("--candidate-report", type=Path); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "baseline": evaluate_frozen(args.workspace / "outputs" / BASELINE_RUN, args.output)
    elif args.command == "coverage-pilot":
        run = run_coverage_pilot(args.workspace, args.run_name, args.steps, args.train_images, args.resume); evaluate_frozen(run, run / "m3_evaluation")
    elif args.command == "compare":
        compare_reports(args.workspace / "outputs" / "m3_b1_frozen_baseline", args.workspace / "outputs" / args.run_name / "m3_evaluation", args.workspace / "outputs" / "m3_comparison.json")
    elif args.command == "pairwise":
        paired_visuals(args.workspace, args.run_name)
    elif args.command == "evaluate":
        workspace = args.workspace.resolve(); run = _safe_run(workspace, args.run_name); output = _new_output(workspace, args.output); evaluate_frozen(run, output, args.checkpoint)
    else:
        workspace = args.workspace.resolve()
        if args.baseline_report is None or args.candidate_report is None: raise ValueError("pairwise-runs requires --baseline-report and --candidate-report")
        output = _new_output(workspace, args.output)
        compare_completed_runs(workspace, args.baseline_report.resolve(), args.candidate_report.resolve(), output)


if __name__ == "__main__": main()
