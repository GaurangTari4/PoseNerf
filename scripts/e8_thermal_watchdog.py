"""Independent read-only GPU watchdog that requests a checkpointed E8 stop."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import csv
from datetime import datetime, timezone
from pathlib import Path


def gpu_state() -> dict[str, float]:
    output = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
    temperature, used, total, utilization, power = (float(value.strip()) for value in output.split(","))
    return {"temperature_c": temperature, "memory_used_mb": used, "memory_total_mb": total, "utilization_percent": utilization, "power_watts": power}


def _status_age(run_dir: Path) -> float | None:
    try:
        return max(0.0, time.time() - float(json.loads((run_dir / "live" / "status.json").read_text(encoding="utf-8"))["timestamp"]))
    except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def append_gpu_csv(run_dir: Path, event: dict) -> None:
    path = run_dir / "e8_logs" / "gpu_telemetry.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp_utc", "event", "temperature_c", "utilization_percent", "power_watts", "memory_used_mb", "memory_total_mb", "thermal_event"]
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new: writer.writeheader()
        writer.writerow(event)


def thermal_decision(temperature_c: float, critical_samples: int, *, warning_c: float, critical_c: float, emergency_c: float) -> tuple[str | None, int]:
    """Return (event/reason, updated sustained-critical counter)."""
    if temperature_c >= emergency_c:
        return "emergency_temperature", critical_samples
    if temperature_c >= critical_c:
        critical_samples += 1
        return ("critical_temperature" if critical_samples >= 3 else "critical_warning"), critical_samples
    if temperature_c >= warning_c:
        return "warning_temperature", 0
    return None, 0


def acknowledged(run_dir: Path) -> bool:
    try:
        return json.loads((run_dir / "live" / "status.json").read_text(encoding="utf-8")).get("state") in {"stopped", "completed", "failed"}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False


def escalate_verified_trainer(lock_path: Path, run_dir: Path, timeout: float) -> bool:
    """Kill only the launcher-recorded trainer after it misses the stop deadline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if acknowledged(run_dir): return False
        time.sleep(1)
    try:
        fields = dict(line.split("=", 1) for line in lock_path.read_text(encoding="utf-8").splitlines() if "=" in line)
        if fields.get("run") != run_dir.name or not fields.get("trainer_pid", "").isdigit(): return False
        subprocess.run(["taskkill", "/PID", fields["trainer_pid"], "/T", "/F"], check=True, capture_output=True, text=True, timeout=10)
        return True
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="E8 thermal policy: warn 85C, sustained stop 88C, immediate emergency stop 90C.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--warning-temperature-c", type=float, default=85.0)
    parser.add_argument("--critical-temperature-c", type=float, default=88.0)
    parser.add_argument("--emergency-temperature-c", type=float, default=90.0)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--stale-seconds", type=float, default=120.0)
    parser.add_argument("--trainer-lock", type=Path, help="Verified E8 launcher lock used only for bounded post-request escalation.")
    parser.add_argument("--escalation-seconds", type=float, default=45.0)
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1 or args.samples < 1 or args.interval <= 0 or args.stale_seconds <= 0 or not args.warning_temperature_c < args.critical_temperature_c < args.emergency_temperature_c: raise ValueError("invalid watchdog threshold")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    request = args.run_dir / "graceful_stop.request"; log = args.run_dir / "thermal_watchdog.jsonl"; hot = 0; critical = 0
    while not request.exists():
        event = {"event": "heartbeat", "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        try:
            state = gpu_state(); event.update(state); event["telemetry_age_seconds"] = _status_age(args.run_dir)
            thermal_event, critical = thermal_decision(state["temperature_c"], critical, warning_c=args.warning_temperature_c, critical_c=args.critical_temperature_c, emergency_c=args.emergency_temperature_c)
            event["thermal_event"] = thermal_event; event["critical_samples"] = critical
            event["over_temperature"] = state["temperature_c"] >= args.critical_temperature_c
            event["over_memory"] = state["memory_used_mb"] / state["memory_total_mb"] >= args.memory_fraction
            event["telemetry_stale"] = event["telemetry_age_seconds"] is not None and event["telemetry_age_seconds"] > args.stale_seconds
            hot = hot + 1 if event["over_memory"] or event["telemetry_stale"] else 0; event["consecutive_hot_samples"] = hot
            with log.open("a", encoding="utf-8") as handle: handle.write(json.dumps(event, sort_keys=True) + "\n"); handle.flush()
            append_gpu_csv(args.run_dir, event)
            if thermal_event in {"critical_temperature", "emergency_temperature"} or hot >= args.samples:
                reason = thermal_event if thermal_event in {"critical_temperature", "emergency_temperature"} else ("stale_telemetry" if event["telemetry_stale"] else "vram_threshold")
                request.write_text(json.dumps({"reason": reason, "state": event}, indent=2), encoding="utf-8")
                with log.open("a", encoding="utf-8") as handle: handle.write(json.dumps({"event": "stop_requested", "reason": reason, "timestamp_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
                if args.trainer_lock and escalate_verified_trainer(args.trainer_lock, args.run_dir, args.escalation_seconds):
                    with log.open("a", encoding="utf-8") as handle: handle.write(json.dumps({"event": "trainer_terminated_after_unacknowledged_stop", "timestamp_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
                return
        except (OSError, subprocess.SubprocessError, IndexError, ValueError) as error:
            with log.open("a", encoding="utf-8") as handle: handle.write(json.dumps({"event": "sensor_error", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "error": str(error)}) + "\n")
        time.sleep(args.interval)


if __name__ == "__main__": main()
