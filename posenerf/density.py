"""Numerically stable, inference-only ray-termination statistics."""
from __future__ import annotations
import torch

def termination_statistics(z_vals: torch.Tensor, weights: torch.Tensor, opacity_epsilon: float = 1e-4) -> dict[str, torch.Tensor]:
    """Compute statistics conditional on foreground termination mass.

    Background mass (1-opacity) is explicit and does not create a fictitious
    far-plane depth. Low-opacity rays are flagged rather than normalized.
    """
    opacity = weights.sum(-1)
    safe = opacity.clamp_min(opacity_epsilon)
    probability = weights / safe[..., None]
    expected = (probability * z_vals).sum(-1)
    variance = (probability * (z_vals - expected[..., None]).square()).sum(-1)
    entropy = -(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()).sum(-1) / torch.log(torch.tensor(float(max(z_vals.shape[-1], 2)), device=z_vals.device))
    peak_index = weights.argmax(-1); peak_depth = z_vals.gather(-1, peak_index[..., None]).squeeze(-1)
    left = torch.nn.functional.pad(weights[..., :-1], (1, 0), value=-float("inf")); right = torch.nn.functional.pad(weights[..., 1:], (0, 1), value=-float("inf"))
    significant = weights >= (weights.max(-1, keepdim=True).values * .10)
    peaks = ((weights >= left) & (weights >= right) & significant).sum(-1)
    return {"opacity": opacity, "background_weight": (1-opacity).clamp_min(0), "expected_depth": expected, "peak_depth": peak_depth, "variance": variance, "entropy": entropy, "peak_count": peaks, "low_opacity": opacity < opacity_epsilon}
