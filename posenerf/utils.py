from __future__ import annotations

import json
import random
import ast
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def safe_torch_load(path: Path, device: torch.device) -> dict:
    """Load tensor-only checkpoints without permitting arbitrary pickle execution."""
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "PoseNeRF requires a PyTorch version supporting torch.load(weights_only=True) for safe checkpoint loading."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {path} must contain a dictionary payload.")
    return payload


def capture_rng_state() -> dict:
    np_state = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": repr(random.getstate()),
        "numpy": {"name": np_state[0], "keys": np_state[1].tolist(), "pos": np_state[2], "has_gauss": np_state[3], "cached_gaussian": np_state[4]},
    }


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    # Checkpoints may have been loaded with map_location="cuda" even though
    # the CPU generator states must remain CPU ByteTensors.
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])
    if state.get("python"):
        random.setstate(ast.literal_eval(state["python"]))
    if state.get("numpy"):
        item = state["numpy"]
        np.random.set_state((item["name"], np.array(item["keys"], dtype=np.uint32), item["pos"], item["has_gauss"], item["cached_gaussian"]))


def maybe_empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
