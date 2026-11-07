"""Safe discovery of renderable PoseNeRF checkpoints."""
from __future__ import annotations

from pathlib import Path
import torch

from .utils import safe_torch_load


def discover_checkpoints(run: Path, device: torch.device) -> list[dict]:
    root = run / "checkpoints"
    result = []
    for path in sorted(root.glob("*.pth")) if root.is_dir() else ():
        try:
            payload = safe_torch_load(path, device)
            state, config = payload.get("model_state_dict"), payload.get("config")
            if not isinstance(state, dict) or not isinstance(config, dict):
                continue
            kind = "final" if path.name.endswith("_trained.pth") else "best" if path.name.endswith("_best.pth") else "resume" if "resume" in path.name else "versioned"
            # Final artifacts contain the best-validation step in ``step``;
            # their model state is nevertheless the completed training state.
            step = int(config.get("train", {}).get("num_steps", -1)) if kind == "final" else int(payload.get("step", -1))
            result.append({"id": path.name, "path": path, "kind": kind, "step": step, "config": config, "signature": payload.get("config_signature", {})})
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            continue
    return result


def resolve_checkpoint(run: Path, checkpoint_id: str | None, device: torch.device) -> dict:
    choices = discover_checkpoints(run, device)
    if not choices:
        raise ValueError("selected experiment has no renderable checkpoint")
    if checkpoint_id:
        choice = next((item for item in choices if item["id"] == checkpoint_id), None)
        if choice is None:
            raise ValueError("selected rendering checkpoint is unavailable")
        return choice
    return next((item for item in choices if item["kind"] == "final"), choices[0])
