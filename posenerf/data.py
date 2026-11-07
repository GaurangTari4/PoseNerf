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

from .config import CHESS_ARCHIVE_SIZE_BYTES, CHESS_URL, ExperimentConfig, PathsConfig


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
    val_frames: List[FrameRecord]
    test_frames: List[FrameRecord]
    train_images: torch.Tensor
    train_depths: torch.Tensor
    train_masks: torch.Tensor
    train_poses: torch.Tensor
    val_images: torch.Tensor
    val_depths: torch.Tensor
    val_masks: torch.Tensor
    val_poses: torch.Tensor
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
        _extract_sequence_archives(paths.scene_dir)
        return paths.scene_dir
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    if not target_zip.exists() or target_zip.stat().st_size != CHESS_ARCHIVE_SIZE_BYTES:
        _download_with_resume(CHESS_URL, target_zip, CHESS_ARCHIVE_SIZE_BYTES)
    if target_zip.stat().st_size != CHESS_ARCHIVE_SIZE_BYTES:
        raise ValueError(
            f"Unexpected Chess archive size: {target_zip.stat().st_size} bytes; "
            f"expected {CHESS_ARCHIVE_SIZE_BYTES}."
        )
    with zipfile.ZipFile(target_zip, "r") as archive:
        if archive.testzip() is not None:
            raise zipfile.BadZipFile("Chess archive failed CRC validation.")
        _safe_extract_zip(archive, paths.data_dir)
    _extract_sequence_archives(paths.scene_dir)
    return paths.scene_dir


def _download_with_resume(url: str, destination: Path, expected_size: int) -> None:
    """Resume an interrupted HTTP range download and verify its exact size."""
    existing_size = destination.stat().st_size if destination.exists() else 0
    if existing_size > expected_size:
        raise ValueError(f"Existing archive exceeds expected size: {existing_size} > {expected_size}.")
    if existing_size == expected_size:
        return
    request = urllib.request.Request(url)
    if existing_size:
        request.add_header("Range", f"bytes={existing_size}-")
    with urllib.request.urlopen(request) as response:
        status = response.getcode()
        if existing_size and status != 206:
            raise RuntimeError("Server did not honor the requested range; refusing to append to the partial archive.")
        mode = "ab" if existing_size else "wb"
        with destination.open(mode) as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
    if destination.stat().st_size != expected_size:
        raise RuntimeError(
            f"Incomplete Chess archive: {destination.stat().st_size} bytes; expected {expected_size}. Re-run to resume."
        )


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract a trusted archive while rejecting traversal and symlink entries."""
    root = destination.resolve()
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Archive member escapes destination: {member.filename}")
        mode = member.external_attr >> 16
        if mode and (mode & 0o170000) == 0o120000:
            raise ValueError(f"Archive contains a symbolic link: {member.filename}")
    archive.extractall(root)


def _extract_sequence_archives(scene_dir: Path) -> None:
    """Unpack the official per-sequence archives required by the frame loader."""
    for sequence_archive in sorted(scene_dir.glob("seq-*.zip")):
        sequence_dir = scene_dir / sequence_archive.stem
        if sequence_dir.exists():
            continue
        with zipfile.ZipFile(sequence_archive) as archive:
            if archive.testzip() is not None:
                raise zipfile.BadZipFile(f"Sequence archive failed CRC validation: {sequence_archive}")
            _safe_extract_zip(archive, scene_dir)


def read_split_file(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sequence_name_to_dir(scene_dir: Path, name: str) -> Path:
    return scene_dir / f"seq-{int(name.replace('sequence', '')):02d}"


def load_pose(path: Path) -> np.ndarray:
    pose = np.loadtxt(path).astype(np.float32)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Expected a finite 4x4 camera-to-world pose matrix in {path}.")
    if not np.allclose(pose[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), atol=1e-4):
        raise ValueError(f"Pose matrix in {path} is not a homogeneous transform.")
    return pose


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
    # Microsoft documents 65535 as invalid. Real Chess frames also contain
    # zero-valued holes, which cannot represent a physical positive depth.
    valid = (arr > 0.0) & (arr != 65535.0)
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


def split_train_validation_frames(
    frames: List[FrameRecord], val_fraction: float, max_train_images: int | None, max_val_images: int | None
) -> tuple[List[FrameRecord], List[FrameRecord]]:
    """Create deterministic, disjoint train/validation subsets from official train frames."""
    if len(frames) < 2:
        raise ValueError("At least two official training frames are required for a train/validation split.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1.")
    val_count = max(1, round(len(frames) * val_fraction))
    val_indices = set(np.linspace(0, len(frames) - 1, val_count, dtype=int).tolist())
    val_frames = [frame for idx, frame in enumerate(frames) if idx in val_indices]
    train_frames = [frame for idx, frame in enumerate(frames) if idx not in val_indices]
    if max_train_images is not None:
        train_frames = _subsample_frames(train_frames, max_train_images)
    if max_val_images is not None:
        val_frames = _subsample_frames(val_frames, max_val_images)
    return train_frames, val_frames


def _subsample_frames(frames: List[FrameRecord], max_images: int) -> List[FrameRecord]:
    if len(frames) <= max_images:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_images, dtype=int)
    return [frames[i] for i in indices]


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


def load_chess_dataset(config: ExperimentConfig, paths: PathsConfig, device: torch.device, storage_device: torch.device | None = None, include_test: bool = True) -> LoadedDataset:
    storage_device = storage_device if storage_device is not None else device
    train_split = paths.scene_dir / "TrainSplit.txt"
    test_split = paths.scene_dir / "TestSplit.txt"
    official_train_frames = collect_frames(paths.scene_dir, read_split_file(train_split))
    train_frames, val_frames = split_train_validation_frames(
        official_train_frames,
        config.train.val_fraction,
        config.train.max_train_images,
        config.train.max_val_images,
    )
    test_frames = collect_frames(paths.scene_dir, read_split_file(test_split), config.train.max_test_images) if include_test else []
    train_images, train_depths, train_masks, train_poses = load_frame_records(
        train_frames, config.scene.downsample, storage_device
    )
    val_images, val_depths, val_masks, val_poses = load_frame_records(val_frames, config.scene.downsample, storage_device)
    if test_frames:
        test_images, test_depths, test_masks, test_poses = load_frame_records(test_frames, config.scene.downsample, storage_device)
    else:
        h, w = config.scene.height, config.scene.width
        test_images = torch.empty((0, h, w, 3), device=storage_device)
        test_depths = torch.empty((0, h, w), device=storage_device)
        test_masks = torch.empty((0, h, w), device=storage_device)
        test_poses = torch.empty((0, 4, 4), device=storage_device)
    return LoadedDataset(
        train_frames=train_frames,
        val_frames=val_frames,
        test_frames=test_frames,
        train_images=train_images,
        train_depths=train_depths,
        train_masks=train_masks,
        train_poses=train_poses,
        val_images=val_images,
        val_depths=val_depths,
        val_masks=val_masks,
        val_poses=val_poses,
        test_images=test_images,
        test_depths=test_depths,
        test_masks=test_masks,
        test_poses=test_poses,
    )

