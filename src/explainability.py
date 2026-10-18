"""G4.4 frozen Integrated Gradients and equal-budget temporal masking."""
from __future__ import annotations
import numpy as np
import torch


def integrated_gradients(model, x: torch.Tensor, target: int, steps: int = 32):
    if x.shape != (1, 12, 1000) or not 0 <= target < 5 or steps != 32:
        raise ValueError('G4.4 IG configuration mismatch')
    baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    # Right Riemann approximation, exactly 32 integration steps.
    for k in range(1, steps + 1):
        xi = (baseline + (k / steps) * (x - baseline)).detach().requires_grad_(True)
        score = model(xi)[0, target]
        grad = torch.autograd.grad(score, xi)[0]
        total += grad.detach()
    ig = (x - baseline) * total / steps
    with torch.no_grad():
        actual = float((model(x)[0, target] - model(baseline)[0, target]).item())
        approx = float(ig.sum().item())
    return ig.detach(), {'completeness_delta': actual - approx,
                         'completeness_relative_error': abs(actual - approx) / max(abs(actual), 1e-8),
                         'logit_difference_to_baseline': actual, 'ig_sum': approx}


def best_two_windows(scores: np.ndarray, width: int = 50):
    a = np.asarray(scores, dtype=np.float64)
    if a.shape != (1000,) or width != 50 or not np.isfinite(a).all():
        raise ValueError('Invalid importance vector/window')
    s = np.convolve(a, np.ones(width, dtype=np.float64), mode='valid')
    # For each first start, find best second non-overlapping start. Ties: smallest starts.
    best = None
    for i in range(len(s)):
        for j in range(i + width, len(s)):
            candidate = (-float(s[i] + s[j]), i, j)
            if best is None or candidate < best:
                best = candidate
    return (best[1], best[2])


def random_two_windows(rng: np.random.Generator, width: int = 50):
    # Uniform over ordered valid pairs i<j, using rejection sampling.
    while True:
        a, b = rng.choice(1000 - width + 1, size=2, replace=False)
        a, b = sorted((int(a), int(b)))
        if b >= a + width:
            return (a, b)


def masked(x: torch.Tensor, starts, width: int = 50):
    a, b = map(int, starts)
    if not (0 <= a < b <= 1000 - width and b >= a + width and width == 50):
        raise ValueError('Invalid non-overlapping windows')
    out = x.clone()
    out[..., a:a+width] = 0
    out[..., b:b+width] = 0
    return out
