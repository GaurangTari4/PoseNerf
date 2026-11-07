"""CPU-only scene snapshots for the read-only local dashboard."""
from __future__ import annotations

from typing import Any

import torch


def camera_frustum_vertices(pose_c2w: torch.Tensor, scale: float = 0.15) -> list[list[float]]:
    """Return camera centre plus four image-plane corners in world coordinates."""
    local = torch.tensor([[0., 0., 0.], [-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]]) * scale
    local[0] = 0
    world = (pose_c2w[:3, :3].cpu() @ local.T).T + pose_c2w[:3, 3].cpu()
    return world.tolist()


def scene_snapshot(images: torch.Tensor, depths: torch.Tensor, masks: torch.Tensor, poses: torch.Tensor, ray_dirs_cam: torch.Tensor, max_points: int = 4000) -> dict[str, Any]:
    """Deterministically back-project valid training RGB-D pixels using C2W poses."""
    h, w = images.shape[1:3]
    dirs = ray_dirs_cam.detach().cpu().reshape(-1, 3)
    points: list[dict[str, Any]] = []
    cameras = [pose[:3, 3].tolist() for pose in poses[::max(1, len(poses) // 48)]]
    per_frame = max(1, max_points // max(1, len(images)))
    for image, depth, mask, pose in zip(images, depths, masks, poses):
        valid = (mask.reshape(-1) > 0.5).nonzero(as_tuple=False).flatten()
        if not len(valid):
            continue
        indices = valid[torch.linspace(0, len(valid) - 1, min(per_frame, len(valid))).long()]
        world_dirs = (pose[:3, :3].cpu() @ dirs[indices].T).T
        xyz = pose[:3, 3].cpu() + world_dirs * depth.reshape(-1)[indices].cpu().unsqueeze(1)
        rgb = (image.reshape(-1, 3)[indices].clamp(0, 1) * 255).to(torch.uint8)
        for point, color in zip(xyz, rgb):
            points.append({"position": point.tolist(), "color": "#%02x%02x%02x" % tuple(color.tolist())})
    return {"coordinate_system": "world; camera-to-world 7-Scenes poses", "points": points, "cameras": cameras, "frustums": [camera_frustum_vertices(pose) for pose in poses[::max(1, len(poses) // 24)]]}


def predicted_scene_snapshot(rendered: list[tuple], poses: torch.Tensor, ray_dirs_cam: torch.Tensor, checkpoint_step: int, near: float, far: float, max_points: int = 4000) -> dict[str, Any]:
    """Back-project *only NeRF-rendered* RGB/depth, never observed depth."""
    dirs = ray_dirs_cam.detach().cpu().reshape(-1, 3)
    points: list[dict[str, Any]] = []
    per_frame = max(1, max_points // max(1, len(rendered)))
    for index, _gt_rgb, predicted_rgb, predicted_depth in rendered:
        depth = predicted_depth.detach().cpu().reshape(-1)
        colors = predicted_rgb.detach().cpu().reshape(-1, 3)
        valid = torch.isfinite(depth) & torch.isfinite(colors).all(dim=1) & (depth >= near) & (depth <= far)
        candidates = valid.nonzero(as_tuple=False).flatten()
        if not len(candidates):
            continue
        selected = candidates[torch.linspace(0, len(candidates) - 1, min(per_frame, len(candidates))).long()]
        pose = poses[index].cpu(); world_dirs = (pose[:3, :3] @ dirs[selected].T).T
        xyz = pose[:3, 3] + world_dirs * depth[selected].unsqueeze(1)
        rgb = (colors[selected].clamp(0, 1) * 255).to(torch.uint8)
        points.extend({"position": point.tolist(), "color": "#%02x%02x%02x" % tuple(color.tolist())} for point, color in zip(xyz, rgb))
    return {
        "provenance": {"source": "NeRF-rendered RGB and predicted depth only", "checkpoint_step": checkpoint_step},
        "coordinate_system": "world; camera-to-world 7-Scenes poses",
        "points": points,
        "cameras": [poses[index, :3, 3].tolist() for index, *_ in rendered],
        "frustums": [camera_frustum_vertices(poses[index]) for index, *_ in rendered],
    }
