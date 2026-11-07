"""Read-only matched frozen evaluation for the completed E3/E6/E7/E8 runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posenerf.milestone3 import evaluate_frozen
from posenerf.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", choices=("E3_final", "E6_final", "E7_final", "E8_final"))
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate previously rendered frame batches without rerendering.")
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    selected = (
        ("E3_final", "m3_e3_capacity_8x192_12k"),
        ("E6_final", "m5b_e6_capacity_8x384_12k"),
        ("E7_final", "m12_e7_extended_8x384_48k"),
        ("E8_final", "m13_e8_manual_10x512_48k"),
    )
    if args.aggregate:
        if args.label is not None or args.frame_indices is not None:
            raise ValueError("--aggregate cannot be combined with --label or --frame-indices")
        summaries = {}
        protocol_hashes = set()
        for label, _ in selected:
            reports = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / label).glob("batch_*/baseline_report.json"))]
            if not reports:
                raise FileNotFoundError(f"no completed batches for {label}")
            frames = [frame for report in reports for frame in report["frames"]]
            if len(frames) != 12 or len({frame["frame"] for frame in frames}) != 12:
                raise ValueError(f"{label} does not contain exactly one result for each of 12 frozen frames")
            first = reports[0]
            if any(report["checkpoint_sha256"] != first["checkpoint_sha256"] for report in reports):
                raise ValueError(f"{label} batches use different checkpoints")
            protocol_hashes.add(first["frozen_protocol_sha256"])
            metrics = {key: float(np.mean([frame[key] for frame in frames])) for key in ("psnr", "ssim_global", "depth_mae", "rgb_mse")}
            latencies = [frame["latency_ms"] for frame in frames]
            metrics["render_latency_ms_mean"] = float(np.mean(latencies))
            metrics["render_latency_ms_p95"] = float(np.percentile(latencies, 95))
            metrics["cuda_peak_mb"] = float(max(report["metrics"]["cuda_peak_mb"] for report in reports))
            summaries[label] = {"source_run": first["source_run"], "checkpoint": first["checkpoint"], "checkpoint_sha256": first["checkpoint_sha256"], "checkpoint_step": first["checkpoint_step"], **metrics, "frames": frames}
        if len(protocol_hashes) != 1:
            raise ValueError(f"frozen protocol mismatch: {protocol_hashes}")
        save_json(output / "matched_frozen_comparison.json", {"frozen_protocol_sha256": protocol_hashes.pop(), "evaluation_protocol": "12 frozen validation views at 160x120; 48 coarse + 48 fine samples; include_test=False", "reports": summaries})
        print(json.dumps({label: {key: value for key, value in report.items() if key != "frames"} for label, report in summaries.items()}, indent=2))
        return
    selected = tuple(item for item in selected if args.label is None or item[0] == args.label)
    batch_name = ("batch_" + "_".join(f"{index:02d}" for index in args.frame_indices)) if args.frame_indices else None
    destinations = {label: (output / label / batch_name if batch_name else output / label) for label, _ in selected}
    if any(destination.exists() for destination in destinations.values()):
        raise FileExistsError("refusing to overwrite an existing per-checkpoint evaluation")
    reports = {
        label: evaluate_frozen(workspace / "outputs" / run_name, destinations[label], "hierarchical_nerf_trained.pth", frame_indices=tuple(args.frame_indices) if args.frame_indices else None)
        for label, run_name in selected
    }
    protocol_hashes = {report["frozen_protocol_sha256"] for report in reports.values()}
    if len(protocol_hashes) != 1:
        raise ValueError(f"frozen protocol mismatch: {protocol_hashes}")
    summary = {
        label: {
            "source_run": report["source_run"],
            "checkpoint": report["checkpoint"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "checkpoint_step": report["checkpoint_step"],
            **report["metrics"],
        }
        for label, report in reports.items()
    }
    if args.label is None and args.frame_indices is None:
        save_json(
            output / "matched_frozen_comparison.json",
            {
                "frozen_protocol_sha256": protocol_hashes.pop(),
                "evaluation_protocol": "12 frozen validation views at 160x120; 48 coarse + 48 fine samples; include_test=False",
                "reports": summary,
            },
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
