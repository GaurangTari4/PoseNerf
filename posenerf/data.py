from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .config import CHESS_URL, ExperimentConfig, PathsConfig


@dataclass
class FrameRecord:
    sequence: str
    stem: str
    color_path: Path
    depth_path: Path
    pose_path: Path


@dataclass
class LoadedDataset:
    train_frames: List[FrameRecord]
    test_frames: List[FrameRecord]
    train_images: torch.Tensor
    train_depths: torch.Tensor
    train_masks: torch.Tensor
    train_poses: torch.Tensor
    test_images: torch.Tensor
    test_depths: torch.Tensor
    test_masks: torch.Tensor
    test_poses: torch.Tensor


def download_chess_dataset(paths: PathsConfig, force: bool = False) -> Path:
    target_zip = paths.data_dir / "chess.zip"
    if force and target_zip.exists():
        target_zip.unlink()
    if force and paths.scene_dir.exists():
        shutil.rmtree(paths.scene_dir)
    if paths.scene_dir.exists() and (paths.scene_dir / "TrainSplit.txt").exists():
        return paths.scene_dir
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    if not target_zip.exists():
        urllib.request.urlretrieve(CHESS_URL, target_zip)
    with zipfile.ZipFile(target_zip, "r") as archive:
        archive.extractall(paths.data_dir)
    return paths.scene_dir


def read_split_file(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sequence_name_to_dir(scene_dir: Path, name: str) -> Path:
    return scene_dir / f"seq-{int(name.replace('sequence', '')):02d}"


def load_pose(path: Path) -> np.ndarray:
    return np.loadtxt(path).astype(np.float32)


def load_rgb(path: Path, downsample: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if downsample > 1:
        image = image.resize((image.width // downsample, image.height // downsample), Image.Resampling.LANCZOS)
    return np.asarray(image).astype(np.float32) / 255.0


def load_depth(path: Path, downsample: int) -> tuple[np.ndarray, np.ndarray]:
    depth = Image.open(path)
    if downsample > 1:
        depth = depth.resize((depth.width // downsample, depth.height // downsample), Image.Resampling.NEAREST)
    arr = np.asarray(depth).astype(np.float32)
    valid = arr != 65535.0
    arr = arr / 1000.0
    arr[~valid] = 0.0
    return arr, valid.astype(np.float32)


def collect_frames(scene_dir: Path, sequence_names: Iterable[str], max_images: int | None = None) -> List[FrameRecord]:
    frames: List[FrameRecord] = []
    for seq_name in sequence_names:
        seq_dir = sequence_name_to_dir(scene_dir, seq_name)
        for color_path in sorted(seq_dir.glob("*.color.png")):
            stem = color_path.name.replace(".color.png", "")
            pose_path = seq_dir / f"{stem}.pose.txt"
            depth_path = seq_dir / f"{stem}.depth.png"
            if pose_path.exists() and depth_path.exists():
                frames.append(FrameRecord(seq_name, stem, color_path, depth_path, pose_path))
    if max_images is not None and len(frames) > max_images:
        idx = np.linspace(0, len(frames) - 1, max_images, dtype=int)
        frames = [frames[i] for i in idx]
    return frames


def load_frame_records(records: List[FrameRecord], downsample: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    images, depths, masks, poses = [], [], [], []
    for record in tqdm(records, desc="Loading dataset"):
        images.append(load_rgb(record.color_path, downsample))
        depth, mask = load_depth(record.depth_path, downsample)
        depths.append(depth)
        masks.append(mask)
        poses.append(load_pose(record.pose_path))
    return (
        torch.from_numpy(np.stack(images)).float().to(device),
        torch.from_numpy(np.stack(depths)).float().to(device),
        torch.from_numpy(np.stack(masks)).float().to(device),
        torch.from_numpy(np.stack(poses)).float().to(device),
    )


def load_chess_dataset(config: ExperimentConfig, paths: PathsConfig, device: torch.device) -> LoadedDataset:
    train_split = paths.scene_dir / "TrainSplit.txt"
    test_split = paths.scene_dir / "TestSplit.txt"
    train_frames = collect_frames(paths.scene_dir, read_split_file(train_split), config.train.max_train_images)
    test_frames = collect_frames(paths.scene_dir, read_split_file(test_split), config.train.max_test_images)
    train_images, train_depths, train_masks, train_poses = load_frame_records(
        train_frames, config.scene.downsample, device
    )
    test_images, test_depths, test_masks, test_poses = load_frame_records(
        test_frames, config.scene.downsample, device
    )
    return LoadedDataset(
        train_frames=train_frames,
        test_frames=test_frames,
        train_images=train_images,
        train_depths=train_depths,
        train_masks=train_masks,
        train_poses=train_poses,
        test_images=test_images,
        test_depths=test_depths,
        test_masks=test_masks,
        test_poses=test_poses,
    )

