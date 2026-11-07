"""Training-depth-assisted, RGB-only 2D--3D localization for Milestone 9.

All functions below keep the inference boundary explicit: a query contributes
RGB pixels/descriptors only.  Depth and camera poses occur only in the
training-reference database; validation poses are read after inference solely
to measure errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import ExperimentConfig, build_paths
from .data import LoadedDataset, load_chess_dataset
from .milestone3 import _safe_run, load_run_config
from .pose import retrieve_initial_pose, rotation_error_deg, translation_error_m
from .utils import get_device, save_json, seed_everything
from .model import get_rays, render_rays


FEATURE_METHOD = "SIFT"


def camera_matrix(config: ExperimentConfig) -> np.ndarray:
    """The matrix for the *loaded* (downsampled) RGB/depth image resolution."""
    s = config.scene
    return np.array([[s.fx, 0.0, s.cx], [0.0, s.fy, s.cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def backproject_pixels(pixels_xy: np.ndarray, depths_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """OpenCV +Z camera coordinates from pixel centers and metric depth."""
    p = np.asarray(pixels_xy, dtype=np.float64)
    z = np.asarray(depths_m, dtype=np.float64).reshape(-1)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] != z.size:
        raise ValueError("pixels must be Nx2 and depths must have N entries")
    if not np.isfinite(p).all() or not np.isfinite(z).all() or np.any(z <= 0):
        raise ValueError("back-projection requires finite positive depths")
    return np.column_stack(((p[:, 0] - K[0, 2]) * z / K[0, 0], (p[:, 1] - K[1, 2]) * z / K[1, 1], z))


def c2w_points(points_camera: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Transform OpenCV camera coordinates to world coordinates using c2w."""
    points = np.asarray(points_camera, dtype=np.float64)
    pose = np.asarray(c2w, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("c2w must be a finite 4x4 matrix")
    return points @ pose[:3, :3].T + pose[:3, 3]


def project_world(points_world: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with a c2w pose; returns pixels and +Z depths."""
    world = np.asarray(points_world, dtype=np.float64)
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    cam = world @ w2c[:3, :3].T + w2c[:3, 3]
    if np.any(cam[:, 2] <= 0) or not np.isfinite(cam).all():
        raise ValueError("points are behind the camera or nonfinite")
    pixels = np.column_stack((K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2], K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]))
    return pixels, cam[:, 2]


def valid_rigid_c2w(pose: np.ndarray) -> bool:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
        return False
    R = pose[:3, :3]
    return bool(np.allclose(R.T @ R, np.eye(3), atol=1e-4) and np.isclose(np.linalg.det(R), 1.0, atol=1e-4))


@torch.no_grad()
def render_feature_geometry(model, c2w: torch.Tensor, pixels_xy: np.ndarray, config: ExperimentConfig, device: torch.device, opacity_min: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Render feature-point geometry without ever consulting observed depth.

    ``render_rays`` depth is accumulated distance along normalized rays.  The
    OpenCV back-projector requires camera-axis Z, hence distance is multiplied
    by the normalized camera ray's +Z component before returning it.
    """
    from .model import get_ray_directions
    xy = np.asarray(pixels_xy, dtype=np.float64)
    if not len(xy): return np.empty(0, np.float32), np.empty(0, bool)
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, config.scene.width - 1); y = np.clip(np.rint(xy[:, 1]).astype(int), 0, config.scene.height - 1)
    directions = get_ray_directions(config, device)[torch.as_tensor(y, device=device), torch.as_tensor(x, device=device)]
    origins, rays = get_rays(directions, c2w.to(device)); _, distance, opacity = render_rays(model, origins, rays, config, perturb=False)
    camera_unit_z = directions[:, 2] / torch.linalg.norm(directions, dim=-1)
    z = distance * camera_unit_z
    valid = torch.isfinite(z) & torch.isfinite(opacity) & (z > config.scene.near) & (z < config.scene.far) & (opacity >= opacity_min)
    return z.cpu().numpy().astype(np.float32), valid.cpu().numpy()


def _gray_u8(rgb: torch.Tensor) -> np.ndarray:
    array = rgb.detach().cpu().numpy()
    return cv2.cvtColor(np.clip(array * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def _sift():
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("OpenCV SIFT is unavailable; this experiment requires OpenCV local features")
    return cv2.SIFT_create(nfeatures=500, contrastThreshold=0.01)


@dataclass
class ReferenceDatabase:
    frame_ids: list[str]
    descriptors: list[np.ndarray]
    pixels: list[np.ndarray]
    world_points: list[np.ndarray]
    manifest: dict


def _frame_id(frame, scene: Path) -> str:
    return str(frame.color_path.relative_to(scene)).replace("\\", "/")


def _manifest(config: ExperimentConfig, frame_ids: list[str]) -> dict:
    source = {"algorithm": FEATURE_METHOD, "opencv": cv2.__version__, "max_features": 500, "contrast_threshold": 0.01,
              "camera_matrix": camera_matrix(config).tolist(), "depth": "official training depth, uint16 millimetres, zero/65535 invalid",
              "frames": frame_ids}
    source["sha256"] = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    return source


def build_reference_database(data: LoadedDataset, config: ExperimentConfig, scene: Path, cache: Path) -> ReferenceDatabase:
    """Create/cache a deterministic, training-only descriptor-to-world-point DB."""
    ids = [_frame_id(f, scene) for f in data.train_frames]
    val_ids = {_frame_id(f, scene) for f in data.val_frames}
    if set(ids) & val_ids:
        raise ValueError("frozen validation query leaked into reference database")
    manifest = _manifest(config, ids)
    manifest_path, payload_path = cache / "manifest.json", cache / "reference_geometry.npz"
    if manifest_path.exists() and payload_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("sha256") == manifest["sha256"]:
            packed = np.load(payload_path, allow_pickle=False)
            offsets = packed["offsets"]
            return ReferenceDatabase(ids, [packed["descriptors"][offsets[i]:offsets[i + 1]] for i in range(len(ids))],
                                     [packed["pixels"][offsets[i]:offsets[i + 1]] for i in range(len(ids))],
                                     [packed["world_points"][offsets[i]:offsets[i + 1]] for i in range(len(ids))], previous)
    sift, K = _sift(), camera_matrix(config)
    descriptors, pixels, worlds, offsets = [], [], [], [0]
    for rgb, depth, mask, pose in zip(data.train_images, data.train_depths, data.train_masks, data.train_poses):
        keys, desc = sift.detectAndCompute(_gray_u8(rgb), None)
        if desc is None:
            desc, xy = np.empty((0, 128), np.float32), np.empty((0, 2), np.float64)
        else:
            xy = np.asarray([k.pt for k in keys], dtype=np.float64)
            x = np.clip(np.rint(xy[:, 0]).astype(int), 0, depth.shape[1] - 1); y = np.clip(np.rint(xy[:, 1]).astype(int), 0, depth.shape[0] - 1)
            valid = mask[y, x].numpy().astype(bool) & np.isfinite(depth[y, x].numpy()) & (depth[y, x].numpy() > 0)
            desc, xy, x, y = desc[valid], xy[valid], x[valid], y[valid]
        z = depth[y, x].numpy() if len(xy) else np.empty(0)
        world = c2w_points(backproject_pixels(xy, z, K), pose.numpy()) if len(xy) else np.empty((0, 3), np.float64)
        descriptors.append(desc.astype(np.float32)); pixels.append(xy.astype(np.float32)); worlds.append(world.astype(np.float32)); offsets.append(offsets[-1] + len(xy))
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(payload_path, descriptors=np.concatenate(descriptors), pixels=np.concatenate(pixels), world_points=np.concatenate(worlds), offsets=np.array(offsets, dtype=np.int64))
    manifest.update({"reference_images": len(ids), "valid_3d_points": int(offsets[-1]), "validation_excluded": sorted(val_ids)})
    save_json(manifest_path, manifest)
    return ReferenceDatabase(ids, descriptors, pixels, worlds, manifest)


def match_candidates(query_rgb: torch.Tensor, database: ReferenceDatabase, candidate_indices: list[int], ratio: float = 0.75) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], int]:
    sift = _sift(); keys, desc = sift.detectAndCompute(_gray_u8(query_rgb), None)
    if desc is None:
        return np.empty((0, 2)), np.empty((0, 3)), np.empty((0, 2)), np.empty(0, int), [], 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    query_pixels, worlds, reference_pixels, sources, detail, used_query = [], [], [], [], [], set()
    raw_total = 0
    for index in candidate_indices:
        train_desc = database.descriptors[index]
        if len(train_desc) < 2: continue
        forward = matcher.knnMatch(desc, train_desc, k=2); reverse = matcher.match(train_desc, desc)
        reverse_pairs = {(m.trainIdx, m.queryIdx) for m in reverse}
        keep = [pair[0] for pair in forward if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance and (pair[0].queryIdx, pair[0].trainIdx) in reverse_pairs]
        raw_total += len(forward)
        unique = 0
        for match in sorted(keep, key=lambda item: item.distance):
            if match.queryIdx in used_query: continue
            used_query.add(match.queryIdx); unique += 1
            query_pixels.append(keys[match.queryIdx].pt); worlds.append(database.world_points[index][match.trainIdx]); reference_pixels.append(database.pixels[index][match.trainIdx]); sources.append(index)
        detail.append({"reference_frame_id": database.frame_ids[index], "raw_matches": len(forward), "ratio_mutual_matches": len(keep), "unique_matches": unique})
    return np.asarray(query_pixels, np.float64).reshape(-1, 2), np.asarray(worlds, np.float64).reshape(-1, 3), np.asarray(reference_pixels, np.float64).reshape(-1, 2), np.asarray(sources, int), detail, len(keys)


def pnp_ransac(query_pixels: np.ndarray, world_points: np.ndarray, K: np.ndarray, seed: int = 612, threshold_px: float = 2.5) -> dict:
    if len(query_pixels) < 6: return {"ok": False, "reason": "insufficient_correspondences"}
    cv2.setRNGSeed(seed)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(world_points, query_pixels, K, None, iterationsCount=300, reprojectionError=threshold_px, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6: return {"ok": False, "reason": "ransac_insufficient_inliers"}
    inlier = inliers.reshape(-1); fraction = len(inlier) / len(query_pixels)
    if fraction < .15: return {"ok": False, "reason": "low_inlier_fraction"}
    # Inlier-only iterative refinement is a geometric refinement (A2).
    ok, rvec, tvec = cv2.solvePnP(world_points[inlier], query_pixels[inlier], K, None, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return {"ok": False, "reason": "inlier_refinement_failed"}
    R, _ = cv2.Rodrigues(rvec); c2w = np.eye(4); c2w[:3, :3] = R.T; c2w[:3, 3] = (-R.T @ tvec).reshape(3)
    cam_z = (world_points[inlier] @ R[2].reshape(3, 1) + tvec[2]).reshape(-1)
    projected, _ = cv2.projectPoints(world_points[inlier], rvec, tvec, K, None)
    rmse = float(np.sqrt(np.mean(np.square(projected.reshape(-1, 2) - query_pixels[inlier]))))
    if not valid_rigid_c2w(c2w) or not np.all(cam_z > 0) or not np.isfinite(rmse): return {"ok": False, "reason": "invalid_pose_or_cheirality"}
    return {"ok": True, "pose": c2w.astype(np.float32), "inliers": inlier, "inlier_count": int(len(inlier)), "inlier_fraction": fraction, "reprojection_rmse_px": rmse}


def _save_matches(path: Path, query: torch.Tensor, reference: torch.Tensor, q: np.ndarray, p: np.ndarray, inliers: np.ndarray | None) -> None:
    a, b = _gray_u8(query), _gray_u8(reference); canvas = np.concatenate((a, b), axis=1)
    fig, ax = plt.subplots(figsize=(12, 4)); ax.imshow(canvas, cmap="gray"); offset = a.shape[1]
    selected = np.ones(len(q), bool) if inliers is None else np.isin(np.arange(len(q)), inliers)
    for query_xy, ref_xy, keep in zip(q, p, selected):
        ax.plot([query_xy[0], ref_xy[0] + offset], [query_xy[1], ref_xy[1]], color="lime" if keep else "tomato", linewidth=.45)
    ax.set_axis_off(); ax.set_title("query (left) / retrieved reference (right): green PnP inliers"); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def _metrics(rows: list[dict], key: str) -> dict:
    trans = np.array([r[f"{key}_translation_error_m"] for r in rows]); rot = np.array([r[f"{key}_rotation_error_deg"] for r in rows])
    return {"translation_m": {"median": float(np.median(trans)), "mean": float(np.mean(trans)), "p90": float(np.percentile(trans, 90))}, "rotation_deg": {"median": float(np.median(rot)), "mean": float(np.mean(rot)), "p90": float(np.percentile(rot, 90))}, "success_5cm_5deg": float(np.mean((trans <= .05) & (rot <= 5))), "success_10cm_10deg": float(np.mean((trans <= .1) & (rot <= 10)))}


def evaluate(workspace: Path, run_name: str, output: Path, cache: Path, limit: int | None = None, seed: int = 612) -> dict:
    run = _safe_run(workspace, run_name)
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    config = load_run_config(run); device = get_device(); seed_everything(seed)
    data = load_chess_dataset(config, build_paths(workspace), device, storage_device=torch.device("cpu"), include_test=False)
    protocol = json.loads((run / "frozen_protocol.json").read_text())
    scene = build_paths(workspace).scene_dir; ids = [_frame_id(f, scene) for f in data.val_frames]
    if ids != [item.replace("\\", "/") for item in protocol["validation_frames"]]: raise ValueError("frozen validation protocol drift")
    database = build_reference_database(data, config, scene, cache)
    rows = []; K = camera_matrix(config); count = len(ids) if limit is None else min(limit, len(ids))
    for i in range(count):
        started = time.perf_counter(); query = data.val_images[i]; retrieval, nearest = retrieve_initial_pose(query.to(device), data.train_images.to(device), data.train_poses.to(device))
        rgb_errors = (data.train_images - query.unsqueeze(0)).square().mean(dim=(1, 2, 3)).numpy(); candidates = np.argsort(rgb_errors)[:5].tolist()
        pixels, worlds, reference_pixels, match_sources, match_details, features = match_candidates(query, database, candidates)
        result = pnp_ransac(pixels, worlds, K, seed + i)
        fallback = not result["ok"]; estimated = retrieval.cpu() if fallback else torch.from_numpy(result["pose"])
        gt = data.val_poses[i]  # Evaluation only, after all retrieval/matching/PnP decisions.
        row = {"query_frame_id": ids[i], "retrieval_frame_id": database.frame_ids[nearest], "candidate_reference_ids": [database.frame_ids[x] for x in candidates], "query_features": features, "raw_matches": int(sum(x["raw_matches"] for x in match_details)), "filtered_matches": int(len(pixels)), "match_details": match_details, "retrieval_pose": retrieval.cpu().tolist(), "pnp_pose": None if fallback else result["pose"].tolist(), "estimated_pose": estimated.tolist(), "fallback_used": fallback, "failure_reason": result.get("reason"), "inlier_count": int(result.get("inlier_count", 0)), "inlier_fraction": float(result.get("inlier_fraction", 0.0)), "reprojection_rmse_px": result.get("reprojection_rmse_px"), "retrieval_translation_error_m": translation_error_m(retrieval.cpu(), gt), "retrieval_rotation_error_deg": rotation_error_deg(retrieval.cpu(), gt), "pnp_translation_error_m": translation_error_m(estimated, gt), "pnp_rotation_error_deg": rotation_error_deg(estimated, gt), "runtime_seconds": time.perf_counter() - started}
        rows.append(row)
        if i == 0:
            first = candidates[0]; keep = match_sources == first
            inlier = result.get("inliers"); inlier = None if inlier is None else np.flatnonzero(np.isin(np.flatnonzero(keep), inlier))
            _save_matches(output / "smoke_matches.png", query, data.train_images[first], pixels[keep], reference_pixels[keep], inlier)
    report = {"method": "A1/A2 training-depth-assisted SIFT + mutual-ratio matching + RANSAC-PnP + inlier-only iterative geometric refinement", "run": run_name, "protocol_sha256": protocol["protocol_sha256"], "seed": seed, "query_inference_contract": "query RGB only; query pose/depth excluded", "geometry_provenance": database.manifest, "intrinsics": K.tolist(), "camera_convention": "OpenCV +Z; dataset poses are c2w; solvePnP w2c output is inverted", "depth_registration": "RGB/depth are sampled at matching rounded pixels under the existing dataset registration assumption; no sub-pixel registration is claimed", "reference_images": len(database.frame_ids), "pnp_success_count": sum(not r["fallback_used"] for r in rows), "fallback_count": sum(r["fallback_used"] for r in rows), "retrieval": _metrics(rows, "retrieval"), "pnp_or_fallback": _metrics(rows, "pnp"), "frames": rows}
    save_json(output / "localization_report.json", report); return report


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--workspace", type=Path, default=Path("workspace")); p.add_argument("--run-name", default="m3_e3_capacity_8x192_12k"); p.add_argument("--output", type=Path, required=True); p.add_argument("--cache", type=Path, required=True); p.add_argument("--limit", type=int); a = p.parse_args()
    print(json.dumps(evaluate(a.workspace.resolve(), a.run_name, a.output.resolve(), a.cache.resolve(), a.limit), indent=2))


if __name__ == "__main__": main()
