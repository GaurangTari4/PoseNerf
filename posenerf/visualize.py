from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch


def _img(x: torch.Tensor):
    return x.detach().cpu().numpy()


def save_eval_figure(rendered: Sequence[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]], metrics, avg_psnr: float, avg_depth_mae: float, path: Path) -> None:
    fig, axes = plt.subplots(len(rendered), 3, figsize=(16, 4 * len(rendered)))
    if len(rendered) == 1:
        axes = axes[None, :]
    for row, (frame_idx, gt_rgb, pred_rgb, pred_depth) in enumerate(rendered):
        frame_metrics = metrics[row]
        axes[row, 0].imshow(_img(gt_rgb))
        axes[row, 0].set_title(f"Ground Truth RGB #{frame_idx}")
        axes[row, 1].imshow(_img(pred_rgb))
        axes[row, 1].set_title(f"Hierarchical Render #{frame_idx}\nPSNR={frame_metrics['psnr']:.2f} dB")
        axes[row, 2].imshow(_img(pred_depth), cmap="magma")
        axes[row, 2].set_title(f"Hierarchical Depth #{frame_idx}\nMAE={frame_metrics['depth_mae']:.3f} m")
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(f"Average PSNR={avg_psnr:.2f} dB | Average Depth MAE={avg_depth_mae:.3f} m", fontsize=14)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_pose_figure(pose_summaries: Sequence[dict], path: Path) -> None:
    fig, axes = plt.subplots(len(pose_summaries), 4, figsize=(20, 4 * len(pose_summaries)))
    if len(pose_summaries) == 1:
        axes = axes[None, :]
    for row, summary in enumerate(pose_summaries):
        axes[row, 0].imshow(_img(summary["target_rgb"]))
        axes[row, 0].set_title(f"Observed Test RGB #{summary['index']}")
        axes[row, 1].imshow(_img(summary["init_render"]))
        axes[row, 1].set_title(f"Initial Pose #{summary['index']}\nPSNR={summary['init_psnr']:.2f} dB\nrot={summary['init_rot_err']:.2f} deg\ntrans={summary['init_trans_err']:.3f} m")
        axes[row, 2].imshow(_img(summary["est_render"]))
        axes[row, 2].set_title(f"Estimated Pose #{summary['index']}\nPSNR={summary['est_psnr']:.2f} dB\nrot={summary['est_rot_err']:.2f} deg\ntrans={summary['est_trans_err']:.3f} m")
        axes[row, 3].imshow(_img(summary["est_depth"]), cmap="magma")
        axes[row, 3].set_title(f"Estimated Depth #{summary['index']}")
        for ax in axes[row]:
            ax.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_loss_plot(histories: Sequence[dict], path: Path) -> None:
    plt.figure(figsize=(10, 4))
    for summary in histories:
        plt.plot(summary["loss_history"], label=f"frame {summary['index']}")
    plt.title("Pose Optimization Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def save_refinement_figure(before_stats: Sequence[dict], after_stats: Sequence[dict], before_psnr: float, after_psnr: float, after_depth_mae: float, path: Path) -> None:
    fig, axes = plt.subplots(len(after_stats), 4, figsize=(20, 4 * len(after_stats)))
    if len(after_stats) == 1:
        axes = axes[None, :]
    for row, stats in enumerate(after_stats):
        before_row = before_stats[row]
        axes[row, 0].imshow(_img(stats["gt_rgb"]))
        axes[row, 0].set_title(f"Ground Truth RGB #{stats['index']}")
        axes[row, 1].imshow(_img(before_row["pred_rgb"]))
        axes[row, 1].set_title(f"Before Refinement #{stats['index']}\nPSNR={before_row['psnr']:.2f} dB")
        axes[row, 2].imshow(_img(stats["pred_rgb"]))
        axes[row, 2].set_title(f"After Refinement #{stats['index']}\nPSNR={stats['psnr']:.2f} dB")
        axes[row, 3].imshow(_img(stats["pred_depth"]), cmap="magma")
        axes[row, 3].set_title(f"Refined Depth #{stats['index']}\nMAE={stats['depth_mae']:.3f} m")
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(f"Before Avg PSNR={before_psnr:.2f} dB | After Avg PSNR={after_psnr:.2f} dB | After Avg Depth MAE={after_depth_mae:.3f} m", fontsize=14)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)

