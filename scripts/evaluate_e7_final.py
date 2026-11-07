"""Read-only, matched 12-view evaluation for E3, E6, and E7 checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from posenerf.milestone3 import evaluate_frozen
from posenerf.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); workspace = args.workspace.resolve(); output = args.output.resolve()
    selected = [
        ("E3_final", "m3_e3_capacity_8x192_12k", "hierarchical_nerf_trained.pth"),
        ("E6_final", "m5b_e6_capacity_8x384_12k", "hierarchical_nerf_trained.pth"),
        *[(f"E7_{step // 1000}k", "m12_e7_extended_8x384_48k", f"hierarchical_nerf_step_{step:05d}.pth") for step in (24000, 28000, 32000, 36000, 40000, 44000, 48000)],
        ("E7_final", "m12_e7_extended_8x384_48k", "hierarchical_nerf_trained.pth"),
    ]
    reports = {}
    for label, run_name, checkpoint in selected:
        run = workspace / "outputs" / run_name
        report_path = output / label / "baseline_report.json"
        reports[label] = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else evaluate_frozen(run, output / label, checkpoint)
    protocol_hashes = {report["frozen_protocol_sha256"] for report in reports.values()}
    if len(protocol_hashes) != 1: raise ValueError(f"protocol mismatch: {protocol_hashes}")
    summary = {label: {"checkpoint": report["checkpoint"], "checkpoint_sha256": report["checkpoint_sha256"], "checkpoint_step": report["checkpoint_step"], **report["metrics"]} for label, report in reports.items()}
    best_rgb = max(summary, key=lambda label: summary[label]["psnr"])
    best_depth = min(summary, key=lambda label: summary[label]["depth_mae"])
    save_json(output / "e7_final_comparison.json", {"frozen_protocol_sha256": protocol_hashes.pop(), "reports": reports, "summary": summary, "best_rgb_fidelity": best_rgb, "best_depth_accuracy": best_depth, "notes": ["Read-only evaluation; official test frames were excluded.", "All checkpoints were rendered at the same 160x120 frozen 12-view protocol.", "Each report contains per-frame metrics and an identical-viewpoint RGB/depth comparison figure."]})


if __name__ == "__main__": main()
