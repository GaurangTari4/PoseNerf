"""Atomic, read-only training telemetry for the local reconstruction viewer.

The trainer is the sole writer.  Readers must tolerate a run not having
published its first event yet; JSON snapshots are replaced atomically and the
event stream is append-only JSONL.
"""
from __future__ import annotations

import json
import os
import time
import uuid
import csv
from pathlib import Path
from typing import Any


EVENT_REQUIRED = {"event", "step", "timestamp"}
_REPLACE_RETRIES = 6
_REPLACE_INITIAL_BACKOFF_SECONDS = 0.025


def _is_transient_windows_replace_error(error: OSError) -> bool:
    """Return true only for Windows sharing/access errors observed from readers."""
    return getattr(error, "winerror", None) in {5, 32, 33}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique sibling prevents competing writers from sharing one .tmp name.
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        first_transient: OSError | None = None
        for attempt in range(_REPLACE_RETRIES):
            try:
                os.replace(temporary, path)
                return
            except OSError as error:
                if not _is_transient_windows_replace_error(error):
                    raise
                first_transient = first_transient or error
                if attempt + 1 == _REPLACE_RETRIES:
                    raise first_transient
                time.sleep(_REPLACE_INITIAL_BACKOFF_SECONDS * (2**attempt))
    finally:
        # Only remove the unique temporary this invocation created; the last
        # published status remains untouched even after exhausted retries.
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Publish a checkpoint only after its complete temporary file is written."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class RunTelemetry:
    def __init__(self, directory: Path, *, run_id: str = "default", audit_dir: Path | None = None) -> None:
        self.directory = directory
        self.run_id = run_id
        self.events_path = directory / "events.jsonl"
        self.status_path = directory / "status.json"
        self.audit_dir = audit_dir
        self.directory.mkdir(parents=True, exist_ok=True)
        if audit_dir:
            audit_dir.mkdir(parents=True, exist_ok=True)

    def publish(self, event: str, step: int, **fields: Any) -> dict[str, Any]:
        payload = {"event": event, "step": int(step), "timestamp": time.time(), "run_id": self.run_id, **fields}
        if not EVENT_REQUIRED.issubset(payload):
            raise ValueError("telemetry event is missing required schema fields")
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_write_json(self.status_path, payload)
        if self.audit_dir:
            with (self.audit_dir / "training_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
            with (self.audit_dir / "training.log").open("a", encoding="utf-8") as handle:
                handle.write(f"{payload['timestamp']:.6f} {event} step={step}\n")
            if event in {"train_step", "validation"}:
                target = self.audit_dir / "training_metrics.csv"; fields = ["timestamp","event","step","loss","fine_rgb_loss","coarse_rgb_loss","fine_depth_loss","coarse_depth_loss","lr","validation_psnr","validation_depth_mae"]
                new = not target.exists();
                with target.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                    if new: writer.writeheader()
                    writer.writerow(payload)
            if any(key in payload for key in ("cuda_allocated_mb", "cuda_peak_allocated_mb")):
                target = self.audit_dir / "cuda_memory.csv"; fields = ["timestamp","event","step","cuda_allocated_mb","cuda_reserved_mb","cuda_peak_allocated_mb","cuda_peak_reserved_mb"]
                new = not target.exists();
                with target.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                    if new: writer.writeheader()
                    writer.writerow(payload)
        return payload

    def status(self, state: str, step: int, **fields: Any) -> dict[str, Any]:
        if state not in {"running", "paused", "stopped", "completed", "failed"}:
            raise ValueError(f"invalid live-training state: {state}")
        return self.publish("status", step, state=state, **fields)


def read_latest(directory: Path) -> dict[str, Any] | None:
    """Read a complete status publication; return None while a run is absent."""
    path = directory / "status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
