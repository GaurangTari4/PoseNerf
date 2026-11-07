"""Controlled, resumable Stage C launcher (one milestone per invocation)."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import ExperimentConfig, build_paths, ensure_directories
from .data import load_chess_dataset
from .model import HierarchicalNeRF, get_ray_directions
from .training import train_model
from .utils import get_device, save_json, seed_everything, safe_torch_load

def _apply_config(value, source: dict) -> None:
    for name, incoming in source.items():
        if not hasattr(value, name): continue
        current = getattr(value, name)
        if hasattr(current, "__dataclass_fields__") and isinstance(incoming, dict): _apply_config(current, incoming)
        else: setattr(value, name, incoming)


def stage_paths(project: Path, run_name: str):
    base = build_paths(project)
    output = base.outputs_dir / run_name
    return replace(base, outputs_dir=output, checkpoints_dir=output / "checkpoints", visualizations_dir=output / "visualizations", pose_dir=output / "pose_estimation", refinement_dir=output / "refinement")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one bounded PoseNeRF Stage C milestone.")
    parser.add_argument("--project-dir", type=Path, default=Path("workspace"))
    parser.add_argument("--run-name", default="stage_c_chess_160")
    parser.add_argument("--steps", type=int, choices=[500, 1500, 3000, 6000, 12000, 48000], default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--depth-loss-weight", type=float, default=0.1, help="Controlled supervision ablation; zero is RGB-only.")
    parser.add_argument("--train-images", type=int, default=96, help="Number of distinct official-training frames; use only for a coverage ablation.")
    parser.add_argument("--batch-rays", type=int, default=1024, help="Rays per microbatch.")
    parser.add_argument("--downsample", type=int, choices=(2, 4), default=4, help="2 is 320x240; 4 is the 160x120 baseline.")
    parser.add_argument("--hidden", type=int, default=128, help="NeRF width; change only for capacity ablations.")
    parser.add_argument("--layers", type=int, default=6, help="NeRF depth; change only for capacity ablations.")
    parser.add_argument("--coarse-samples", type=int, default=48, help="Hierarchical coarse samples.")
    parser.add_argument("--fine-samples", type=int, default=48, help="Hierarchical fine samples.")
    parser.add_argument("--pos-freqs", type=int, default=10, help="Position Fourier frequencies; recorded in checkpoint metadata.")
    parser.add_argument("--dir-freqs", type=int, default=4, help="Direction Fourier frequencies; recorded in checkpoint metadata.")
    parser.add_argument("--edge-sampling-fraction", type=float, default=0.0, help="Fraction of rays sampled from deterministic Sobel edge maps.")
    parser.add_argument("--edge-threshold", type=float, default=0.20, help="Per-image normalized Sobel threshold for edge maps.")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Microbatches per optimizer and scheduler update.")
    parser.add_argument("--optimizer-pause-seconds", type=float, default=0.0, help="Optional pause after each optimizer update for thermal pacing.")
    parser.add_argument("--resume-from", type=Path, help="External full-state checkpoint; E7 controlled continuation only.")
    parser.add_argument("--scheduler-policy", choices=("controlled_extension_scheduler",), help="Required explicit E7 scheduler policy.")
    parser.add_argument("--e8-continuation", action="store_true", help="Explicit E8 full-state 500-successful-update continuation.")
    parser.add_argument("--resume-existing-continuation", action="store_true", help="Resume only this run directory's E8 continuation checkpoint.")
    parser.add_argument("--continuation-updates", type=int, default=500)
    parser.add_argument("--continuation-start-lr", type=float, default=4e-5)
    parser.add_argument("--continuation-final-lr", type=float, default=1e-5)
    parser.add_argument("--source-sha256", default="")
    parser.add_argument("--snapshot-steps", default="", help="Comma-separated immutable global checkpoint steps.")
    parser.add_argument("--pilot-steps", type=int, help="Bounded E7 pilot count; retains the 48k scheduler horizon.")
    args = parser.parse_args()
    if args.accumulation_steps < 1 or args.batch_rays < 1 or args.optimizer_pause_seconds < 0: raise ValueError("invalid microbatch, accumulation, or pacing configuration")
    paths = stage_paths(args.project_dir, args.run_name); ensure_directories(paths)
    external_state = None
    if args.resume_existing_continuation and (not args.e8_continuation or args.resume_from): raise ValueError("resume-existing continuation cannot use the immutable source path")
    if args.e8_continuation and not args.resume_from and not args.resume_existing_continuation: raise ValueError("fresh E8 continuation requires --resume-from")
    if args.resume_existing_continuation:
        candidate = paths.checkpoints_dir / "hierarchical_nerf_resume_latest.pth"
        if not candidate.is_file(): raise FileNotFoundError("continuation resume checkpoint is missing")
        args.resume_from = candidate
    if args.resume_from:
        if not args.e8_continuation and (args.steps != 48000 or args.scheduler_policy != "controlled_extension_scheduler"): raise ValueError("external E7 resume requires --steps 48000 and --scheduler-policy controlled_extension_scheduler")
        args.resume_from = args.resume_from.resolve()
        if not args.resume_from.is_file(): raise FileNotFoundError(f"source checkpoint does not exist: {args.resume_from}")
        source_hash = hashlib.sha256(args.resume_from.read_bytes()).hexdigest()
        external_state = safe_torch_load(args.resume_from, torch.device("cpu"))
        is_e7_resume = external_state.get("scheduler_policy") == "controlled_extension_scheduler" and bool(external_state.get("extension_schedule"))
        if args.e8_continuation:
            if args.resume_existing_continuation:
                continuation_state = external_state.get("continuation")
                if not isinstance(continuation_state, dict) or continuation_state.get("source_sha256") != "66523e38054adc67c3e6c1e4650e5ac92aacddc8979ab6577313f667deaab972": raise ValueError("checkpoint is not a verified E8 continuation")
                if float(external_state.get("config",{}).get("train",{}).get("depth_loss_weight",-1)) != args.depth_loss_weight: raise ValueError("depth weight does not match continuation identity")
            elif args.source_sha256.lower() != source_hash.lower() or source_hash.lower() != "66523e38054adc67c3e6c1e4650e5ac92aacddc8979ab6577313f667deaab972": raise ValueError("E8 source SHA-256 mismatch")
            model_cfg = external_state.get("config", {}).get("model", {})
            if model_cfg.get("layers") != 10 or model_cfg.get("hidden") != 512 or external_state.get("scheduler_state_dict",{}).get("last_epoch") != 48000: raise ValueError("source is not the completed E8 10x512 full-state checkpoint")
            if args.continuation_updates != 500 or args.continuation_start_lr <= 0 or args.continuation_final_lr <= 0 or args.continuation_start_lr <= args.continuation_final_lr: raise ValueError("invalid E8 continuation schedule")
        elif not is_e7_resume and int(external_state.get("step", -1)) != 11999: raise ValueError("E7 source must end at global step 11999")
        if not all(k in external_state for k in ("optimizer_state_dict","scheduler_state_dict","scaler_state_dict","rng_state","sampler_generator_state")): raise ValueError("external checkpoint is not full-state")
        source_protocol = args.resume_from.parent.parent / "frozen_protocol.json"
        if not source_protocol.is_file() or json.loads(source_protocol.read_text()).get("protocol_sha256") != "18b92efd0173d343011c7b654ff24975c51de31d72f7ac38d9b0eb0e9b32858b": raise ValueError("source frozen protocol mismatch")
        save_json(paths.outputs_dir / "external_resume_provenance.json", {"source_checkpoint":str(args.resume_from.resolve()),"sha256":source_hash,"source_step":int(external_state["step"]),"original_scheduler":external_state.get("scheduler_state_dict"),"continuation":{"type":"cosine","successful_updates":args.continuation_updates,"start_lr":args.continuation_start_lr,"final_lr":args.continuation_final_lr} if args.e8_continuation else None,"scheduler_policy":args.scheduler_policy})
    config = ExperimentConfig(seed=2027); config.scene.downsample = args.downsample
    if external_state:
        _apply_config(config, external_state["config"])
        if args.e8_continuation:
            if config.model.layers != 10 or config.model.hidden != 512: raise ValueError("E8 source architecture must be 10x512")
        elif config.model.layers != 8 or config.model.hidden != 384: raise ValueError("E7 source architecture must be 8x384")
    if not external_state:
        config.train.max_train_images = args.train_images; config.train.max_val_images = 12; config.train.max_test_images = 12
    # Keep the scheduler horizon fixed across cumulative milestones.  The
    # controlled stop boundary is intentionally excluded from checkpoint
    # compatibility so 500 -> 1,500 resumes restore the exact scheduler.
    if external_state:
        resume_next_step = int(external_state["step"]) + 1
        if args.e8_continuation:
            config.train.num_steps = resume_next_step + args.continuation_updates; config.train.stop_after_step = config.train.num_steps; config.train.depth_loss_weight=args.depth_loss_weight; config.train.resume=True
        else: config.train.num_steps = args.steps; config.train.stop_after_step = (resume_next_step + args.pilot_steps) if args.pilot_steps else args.steps; config.train.resume = True
    else:
        config.train.batch_rays = args.batch_rays; config.train.accumulation_steps = args.accumulation_steps; config.train.optimizer_pause_seconds = args.optimizer_pause_seconds; config.train.num_steps = args.steps if args.steps == 48000 else 12000; config.train.stop_after_step = args.steps; config.train.num_coarse_samples = args.coarse_samples; config.train.num_fine_samples = args.fine_samples
        config.train.val_render_interval = config.train.checkpoint_every = 100; config.train.val_frame_count = 2; config.train.use_amp = True; config.train.on_demand_ray_sampling = True; config.train.resume = args.resume
    if not external_state:
        config.train.depth_loss_weight = args.depth_loss_weight
        config.train.edge_sampling_fraction = args.edge_sampling_fraction; config.train.edge_threshold = args.edge_threshold
    if not external_state:
        config.model.hidden = args.hidden; config.model.layers = args.layers; config.model.pos_freqs = args.pos_freqs; config.model.dir_freqs = args.dir_freqs; config.model.render_chunk = 512
    manifest = paths.outputs_dir / "manifest.json"
    if manifest.exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {paths.outputs_dir}; use --resume only for the same milestone configuration.")
    if not manifest.exists():
        save_json(manifest, {"stage": "C", "created_utc": datetime.now(timezone.utc).isoformat(), "data": "Official Microsoft 7-Scenes Chess; validation is train-derived; test withheld", "config": config.to_dict(), "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "platform": platform.platform()})
    save_json(paths.outputs_dir / "e8_logs" / "run_metadata.json", {"run_name": args.run_name, "created_utc": datetime.now(timezone.utc).isoformat(), "config": config.to_dict(), "checkpoint_schedule": sorted({int(x) for x in args.snapshot_steps.split(',') if x.strip()}), "full_state_checkpoint": "checkpoints/hierarchical_nerf_resume_latest.pth"})
    device = get_device(); seed_everything(config.seed)
    dataset = load_chess_dataset(config, paths, device, storage_device=torch.device("cpu"), include_test=False)
    protocol = paths.outputs_dir / "frozen_protocol.json"
    if not protocol.exists():
        frame_ids = lambda frames: [str(frame.color_path.relative_to(paths.scene_dir)) for frame in frames]
        payload = {"seed": config.seed, "official_test_withheld": True, "train_frames": frame_ids(dataset.train_frames), "validation_frames": frame_ids(dataset.val_frames), "test_frames_not_used_for_selection": frame_ids(dataset.test_frames), "render": {"downsample": config.scene.downsample, "coarse": config.train.num_coarse_samples, "fine": config.train.num_fine_samples, "chunk": config.model.render_chunk}}
        payload["protocol_sha256"] = hashlib.sha256(str(payload).encode()).hexdigest()
        save_json(protocol, payload)
    directions = get_ray_directions(config, device); model = HierarchicalNeRF(config).to(device); baseline = HierarchicalNeRF(config).to(device)
    print(f"Dashboard: python -m posenerf.dashboard --outputs-dir {paths.outputs_dir.parent}")
    snapshots={int(x) for x in args.snapshot_steps.split(',') if x.strip()}
    continuation = {"type":"cosine","successful_updates":args.continuation_updates,"start_lr":args.continuation_start_lr,"final_lr":args.continuation_final_lr,"source_sha256":"66523e38054adc67c3e6c1e4650e5ac92aacddc8979ab6577313f667deaab972","resume_existing":args.resume_existing_continuation} if args.e8_continuation else None
    train_model(model, baseline, dataset, directions, config, paths, device, external_resume=args.resume_from, scheduler_policy=args.scheduler_policy or "native", immutable_steps=snapshots, continuation=continuation)


if __name__ == "__main__": main()
