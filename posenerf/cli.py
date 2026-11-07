from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig, build_paths, ensure_directories
from .data import download_chess_dataset, load_chess_dataset
from .model import HierarchicalNeRF, get_ray_directions
from .pose import run_pose_estimation
from .refine import run_refinement
from .training import evaluate_model, load_checkpoint, save_evaluation_metrics, train_model
from .utils import get_device, seed_everything
from .visualize import save_eval_figure


def _checkpoint_identity(checkpoint: Path) -> str:
    stat = checkpoint.stat()
    return f"{checkpoint.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def build_runtime(args):
    config = ExperimentConfig()
    device = get_device()
    seed_everything(config.seed)
    paths = build_paths(Path(args.project_dir))
    ensure_directories(paths)
    if getattr(args, "download_data", False):
        download_chess_dataset(paths, force=getattr(args, "force_download", False))
    dataset = load_chess_dataset(config, paths, device)
    ray_dirs_cam = get_ray_directions(config, device)
    model = HierarchicalNeRF(config).to(device)
    baseline_model = HierarchicalNeRF(config).to(device)
    return config, device, paths, dataset, ray_dirs_cam, model, baseline_model


def cmd_download(args):
    paths = build_paths(Path(args.project_dir))
    ensure_directories(paths)
    scene_dir = download_chess_dataset(paths, force=args.force_download)
    print(f"Dataset ready at: {scene_dir}")


def cmd_train(args):
    config, device, paths, dataset, ray_dirs_cam, model, baseline_model = build_runtime(args)
    train_model(model, baseline_model, dataset, ray_dirs_cam, config, paths, device)


def cmd_evaluate(args):
    config, device, paths, dataset, ray_dirs_cam, model, baseline_model = build_runtime(args)
    checkpoint = Path(args.checkpoint) if args.checkpoint else paths.checkpoints_dir / "hierarchical_nerf_trained.pth"
    load_checkpoint(model, baseline_model, checkpoint, device, config)
    frame_indices = list(range(min(config.train.val_frame_count, len(dataset.test_images))))
    metrics, rendered, avg_psnr, avg_depth_mae = evaluate_model(
        model, dataset.test_images, dataset.test_depths, dataset.test_masks, dataset.test_poses, ray_dirs_cam, config, frame_indices
    )
    save_eval_figure(rendered, metrics, avg_psnr, avg_depth_mae, paths.visualizations_dir / "checkpoint_eval_multiview.png")
    save_evaluation_metrics(paths, metrics, avg_psnr, avg_depth_mae)


def cmd_post_train(args):
    config, device, paths, dataset, ray_dirs_cam, model, baseline_model = build_runtime(args)
    checkpoint = Path(args.checkpoint) if args.checkpoint else paths.checkpoints_dir / "hierarchical_nerf_trained.pth"
    load_checkpoint(model, baseline_model, checkpoint, device, config)
    frame_indices = list(range(min(config.train.val_frame_count, len(dataset.test_images))))
    metrics, rendered, avg_psnr, avg_depth_mae = evaluate_model(
        model, dataset.test_images, dataset.test_depths, dataset.test_masks, dataset.test_poses, ray_dirs_cam, config, frame_indices
    )
    save_eval_figure(rendered, metrics, avg_psnr, avg_depth_mae, paths.visualizations_dir / "checkpoint_eval_multiview.png")
    save_evaluation_metrics(paths, metrics, avg_psnr, avg_depth_mae)
    pose_summaries = run_pose_estimation(model, dataset, ray_dirs_cam, config, paths, device, checkpoint_id=_checkpoint_identity(checkpoint), use_cache=(not args.no_pose_cache))
    run_refinement(model, baseline_model, dataset, ray_dirs_cam, pose_summaries, config, paths, device)
    metrics, _, avg_psnr, avg_depth_mae = evaluate_model(model, dataset.test_images, dataset.test_depths, dataset.test_masks, dataset.test_poses, ray_dirs_cam, config, frame_indices)
    save_evaluation_metrics(paths, metrics, avg_psnr, avg_depth_mae, "refined_test_metrics.json")


def cmd_full_pipeline(args):
    config, device, paths, dataset, ray_dirs_cam, model, baseline_model = build_runtime(args)
    checkpoint = train_model(model, baseline_model, dataset, ray_dirs_cam, config, paths, device)
    frame_indices = list(range(min(config.train.val_frame_count, len(dataset.test_images))))
    metrics, rendered, avg_psnr, avg_depth_mae = evaluate_model(
        model, dataset.test_images, dataset.test_depths, dataset.test_masks, dataset.test_poses, ray_dirs_cam, config, frame_indices
    )
    save_eval_figure(rendered, metrics, avg_psnr, avg_depth_mae, paths.visualizations_dir / "checkpoint_eval_multiview.png")
    save_evaluation_metrics(paths, metrics, avg_psnr, avg_depth_mae)
    pose_summaries = run_pose_estimation(model, dataset, ray_dirs_cam, config, paths, device, checkpoint_id=_checkpoint_identity(checkpoint), use_cache=False)
    run_refinement(model, baseline_model, dataset, ray_dirs_cam, pose_summaries, config, paths, device)
    metrics, _, avg_psnr, avg_depth_mae = evaluate_model(model, dataset.test_images, dataset.test_depths, dataset.test_masks, dataset.test_poses, ray_dirs_cam, config, frame_indices)
    save_evaluation_metrics(paths, metrics, avg_psnr, avg_depth_mae, "refined_test_metrics.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PoseNeRF: hierarchical NeRF pipeline for camera pose estimation.")
    parser.add_argument("--project-dir", default="./workspace", help="Directory that will hold data and outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="Download and extract the Chess dataset.")
    download.add_argument("--force-download", action="store_true")
    download.set_defaults(func=cmd_download)

    train = subparsers.add_parser("train", help="Train the NeRF.")
    train.add_argument("--download-data", action="store_true")
    train.add_argument("--force-download", action="store_true")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate an existing checkpoint.")
    evaluate.add_argument("--download-data", action="store_true")
    evaluate.add_argument("--force-download", action="store_true")
    evaluate.add_argument("--checkpoint", default="")
    evaluate.set_defaults(func=cmd_evaluate)

    post_train = subparsers.add_parser("post-train", help="Run evaluation, pose estimation, and refinement from a checkpoint.")
    post_train.add_argument("--download-data", action="store_true")
    post_train.add_argument("--force-download", action="store_true")
    post_train.add_argument("--checkpoint", default="")
    post_train.add_argument("--no-pose-cache", action="store_true", help="Force re-running pose estimation instead of using cached pose files.")
    post_train.set_defaults(func=cmd_post_train)

    full_pipeline = subparsers.add_parser("full-pipeline", help="Run the full pipeline from data download to refinement.")
    full_pipeline.add_argument("--download-data", action="store_true")
    full_pipeline.add_argument("--force-download", action="store_true")
    full_pipeline.set_defaults(func=cmd_full_pipeline)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
