"""Read-only terminal dashboard for the separately launched E7 trainer."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path


TARGET = "m12_e7_extended_8x384_48k"
TOTAL_STEPS = 48_000


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def tail_events(path: Path, limit: int = 200) -> list[dict]:
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 256_000))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()][-limit:]
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []


def gpu_metrics() -> dict[str, float | str]:
    command = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"]
    try:
        row = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True).stdout.splitlines()[0]
        util, used, total, temperature, power = [item.strip() for item in row.split(",")]
        return {"util": float(util), "used": float(used), "total": float(total), "temp": float(temperature), "power": float(power)}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {"error": "nvidia-smi unavailable"}


def process_info() -> dict[str, str]:
    """Use CIM when allowed; monitoring still works if Windows denies it."""
    command = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Where-Object {$_.CommandLine -match 'posenerf.stage_c' -and $_.CommandLine -match 'm12_e7_extended_8x384_48k'} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        payload = json.loads(result.stdout or "null")
        if isinstance(payload, list): payload = payload[0] if payload else None
        if payload:
            return {"pid": str(payload["ProcessId"]), "parent": str(payload["ParentProcessId"]), "started": str(payload["CreationDate"]), "command": str(payload["CommandLine"])}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, KeyError):
        pass
    return {"pid": "unknown (CIM access unavailable)", "parent": "unknown", "started": "unknown", "command": "unknown"}


def fmt_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0: return "unknown"
    hours, remainder = divmod(int(seconds), 3600); minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only E7 live monitor; Ctrl+C exits safely.")
    parser.add_argument("--run-dir", type=Path, default=Path("workspace/outputs") / TARGET)
    parser.add_argument("--refresh", type=float, default=5.0)
    args = parser.parse_args(); run_dir = args.run_dir.resolve(); live = run_dir / "live"
    loss_window: deque[float] = deque(maxlen=20); prior_lr: float | None = None; hot_samples = 0
    try:
        while True:
            status = read_json(live / "status.json") or {}; events = tail_events(live / "events.jsonl")
            train = [item for item in events if item.get("event") == "train_step"]
            validation = [item for item in events if item.get("event") == "validation"]
            current = int(status.get("step", 0)); completed = max(0, current - 12_000)
            age = time.time() - float(status.get("timestamp", 0) or 0)
            if train:
                latest = train[-1]; loss_window.extend(float(item["loss"]) for item in train[-20:] if math.isfinite(float(item.get("loss", float("nan")))))
            else: latest = status
            speed = 0.0
            if len(train) >= 2:
                delta_step = train[-1]["step"] - train[0]["step"]; delta_time = train[-1]["timestamp"] - train[0]["timestamp"]
                speed = delta_step / delta_time if delta_time > 0 else 0.0
            lr = float(latest.get("lr", float("nan"))); anomalies = []
            if prior_lr is not None and math.isfinite(lr) and lr > prior_lr + 1e-12: anomalies.append("LR increased")
            prior_lr = lr if math.isfinite(lr) else prior_lr
            values = [latest.get(key, 0.0) for key in ("loss", "coarse_rgb_loss", "fine_rgb_loss", "coarse_depth_loss", "fine_depth_loss")]
            if any(not math.isfinite(float(value)) for value in values): anomalies.append("NaN/Inf loss")
            if age > 120: anomalies.append("telemetry stale >120s")
            gpu = gpu_metrics(); hot_samples = hot_samples + 1 if gpu.get("temp", 0) >= 88 else 0
            if hot_samples >= 3: anomalies.append("GPU temperature >=88C sustained")
            if gpu.get("total", 1) and gpu.get("used", 0) / float(gpu["total"]) > .92: anomalies.append("GPU memory >92%")
            checkpoints = sorted((run_dir / "checkpoints").glob("*.pth"), key=lambda p: p.stat().st_mtime) if (run_dir / "checkpoints").exists() else []
            immutable = [path for path in checkpoints if "step_" in path.name]; latest_ckpt = checkpoints[-1] if checkpoints else None
            next_snapshot = next((step for step in (14000,16000,18000,20000,24000,28000,32000,36000,40000,44000,48000) if step > current), None)
            process = process_info(); remaining = max(0, TOTAL_STEPS - current)
            os.system("cls" if os.name == "nt" else "clear")
            print(f"PoseNeRF E7 READ-ONLY MONITOR  {datetime.now().isoformat(timespec='seconds')}  Ctrl+C exits")
            print(f"\nTRAINING  {current:,}/{TOTAL_STEPS:,} ({100*current/TOTAL_STEPS:.2f}%)  E7 +{completed:,}  remaining {remaining:,}")
            print(f"elapsed {fmt_seconds(float(latest.get('elapsed_seconds', 0)))}  speed {speed:.2f} steps/s  ETA {fmt_seconds(remaining/speed) if speed else 'unknown'}")
            print(f"\nLOSS total={latest.get('loss','?')}  coarseRGB={latest.get('coarse_rgb_loss','?')} fineRGB={latest.get('fine_rgb_loss','?')} coarseDepth={latest.get('coarse_depth_loss','?')} fineDepth={latest.get('fine_depth_loss','?')} mean20={(sum(loss_window)/len(loss_window)) if loss_window else float('nan'):.6f}")
            if validation:
                last_val = validation[-1]; trend = [round(float(item['validation_psnr']),4) for item in validation[-10:]]
                print(f"\nVALIDATION (small E6 validation only; not comparable to 12-view headline) latest step {last_val['step']}: PSNR={last_val['validation_psnr']:.5f}, depthMAE={last_val['validation_depth_mae']:.6f}")
                print("PSNR trend:", trend)
            print(f"\nOPTIMIZATION lr={lr:.9g}  policy=controlled_extension_scheduler  anomalies={', '.join(anomalies) if anomalies else 'none'}")
            print(f"GPU util={gpu.get('util','?')}% VRAM={gpu.get('used','?')}/{gpu.get('total','?')}MB temp={gpu.get('temp','?')}C power={gpu.get('power','?')}W")
            print(f"\nCHECKPOINTS latest={latest_ckpt.name if latest_ckpt else 'none'} immutable={immutable[-1].name if immutable else 'none'} next={next_snapshot or 'none'} manifest={'present' if (run_dir/'checkpoints/immutable_snapshot_manifest.json').exists() else 'not yet'}")
            print(f"PROCESS pid={process['pid']} parent={process['parent']} active={'yes' if age <= 120 else 'unknown/stalled'} telemetry_age={age:.1f}s")
            if anomalies: print("\nWARNING: " + " | ".join(anomalies))
            time.sleep(max(.5, args.refresh))
    except KeyboardInterrupt:
        print("\nMonitor stopped safely; the training process was not changed.")


if __name__ == "__main__": main()
