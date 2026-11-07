"""Render a real PoseNeRF checkpoint along interpolated validation C2W poses.

This script uses the project's existing HierarchicalNeRF renderer.  It never
trains or writes checkpoints; generated media is the only output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from posenerf.data import collect_frames, load_frame_records, read_split_file, split_train_validation_frames
from posenerf.milestone3 import _checkpoint_for, load_run_config
from posenerf.model import HierarchicalNeRF, get_ray_directions, render_image
from posenerf.config import build_paths
from posenerf.utils import get_device, seed_everything


def interpolate_c2w(start: torch.Tensor, end: torch.Tensor, fraction: float) -> torch.Tensor:
    """Interpolate C2W translation and project linear rotation to SO(3)."""
    pose = torch.eye(4, dtype=start.dtype, device=start.device)
    rotation = (1.0 - fraction) * start[:3, :3] + fraction * end[:3, :3]
    u, _, vh = torch.linalg.svd(rotation)
    corrected = u @ vh
    if torch.linalg.det(corrected) < 0:
        u[:, -1] *= -1
        corrected = u @ vh
    pose[:3, :3] = corrected
    pose[:3, 3] = (1.0 - fraction) * start[:3, 3] + fraction * end[:3, 3]
    return pose


def trajectory(poses: torch.Tensor, frames: int) -> list[torch.Tensor]:
    """Traverse validation poses in their frozen order with C2W interpolation."""
    if frames < 2 or len(poses) < 2:
        raise ValueError("at least two poses and two frames are required")
    result: list[torch.Tensor] = []
    for position in torch.linspace(0, len(poses) - 1, frames):
        low = int(torch.floor(position).item()); high = min(low + 1, len(poses) - 1)
        result.append(interpolate_c2w(poses[low], poses[high], float(position - low)))
    return result


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MP4 writer unavailable")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _depth_rgb(depth: np.ndarray) -> np.ndarray:
    low, high = np.percentile(depth, (2, 98)); scaled = np.clip((depth - low) / max(high - low, 1e-8), 0, 1)
    return cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True); parser.add_argument("--checkpoint", default="hierarchical_nerf_continuation_trained.pth")
    parser.add_argument("--output", type=Path, default=Path("docs/media")); parser.add_argument("--frames", type=int, default=18); parser.add_argument("--fps", type=int, default=9)
    args = parser.parse_args(); run, output = args.run.resolve(), args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    config = load_run_config(run); device = get_device(); seed_everything(config.seed)
    paths = build_paths(run.parent.parent)
    train_records = collect_frames(paths.scene_dir, read_split_file(paths.scene_dir / "TrainSplit.txt"))
    _, validation_records = split_train_validation_frames(train_records, config.train.val_fraction, config.train.max_train_images, config.train.max_val_images)
    validation_images, _, _, validation_poses = load_frame_records(validation_records, config.scene.downsample, torch.device("cpu"))
    checkpoint, payload = _checkpoint_for(run, args.checkpoint, config, device)
    model = HierarchicalNeRF(config).to(device); model.load_state_dict(payload["model_state_dict"], strict=True); model.eval(); directions = get_ray_directions(config, device)
    rgb_frames: list[np.ndarray] = []; depth_frames: list[np.ndarray] = []
    for pose in trajectory(validation_poses.to(device), args.frames):
        rgb, depth = render_image(model, pose, directions, config)
        rgb_frames.append((rgb.cpu().numpy() * 255).round().astype(np.uint8)); depth_frames.append(_depth_rgb(depth.cpu().numpy()))
    _write_video(output / "posenerf_demo.mp4", rgb_frames, args.fps)
    import imageio.v2 as imageio
    imageio.mimsave(output / "posenerf_demo.gif", rgb_frames, fps=args.fps)
    imageio.mimsave(output / "posenerf_depth.gif", depth_frames, fps=args.fps)
    reference = validation_images[0].cpu().numpy()
    first_depth = depth_frames[0]
    figure, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=140)
    for axis, image, title in zip(axes, (reference, rgb_frames[0] / 255.0, np.abs(reference - rgb_frames[0] / 255.0), first_depth), ("Reference RGB", "E8-D2 reconstruction", "Absolute RGB residual", "E8-D2 rendered depth")):
        axis.imshow(image); axis.set_title(title); axis.axis("off")
    figure.tight_layout(); figure.savefig(output / "posenerf_comparison.png"); plt.close(figure)
    (output / "posenerf_demo_provenance.json").write_text(json.dumps({"kind": "real_checkpoint_reconstruction", "run": run.name, "checkpoint": str(checkpoint), "continuation_success_updates": payload.get("continuation_success_updates"), "source_checkpoint_step": payload.get("step"), "camera_convention": "OpenCV C2W from frozen train-derived validation poses", "trajectory": "linear translation + SO(3)-projected rotation interpolation", "frames": args.frames, "resolution": [config.scene.width, config.scene.height]}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
