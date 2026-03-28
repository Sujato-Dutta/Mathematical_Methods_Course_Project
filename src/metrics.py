from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    ref = reference.detach().cpu().squeeze().numpy()
    est = estimate.detach().cpu().squeeze().numpy()
    return float(peak_signal_noise_ratio(ref, est, data_range=1.0))


def compute_ssim(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    ref = reference.detach().cpu().squeeze().numpy()
    est = estimate.detach().cpu().squeeze().numpy()
    return float(structural_similarity(ref, est, data_range=1.0))


def summarize_metric_rows(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    if not rows:
        return summary

    numeric_keys = [key for key in rows[0] if isinstance(rows[0][key], (int, float))]
    for key in numeric_keys:
        summary[f"mean_{key}"] = float(np.mean([float(row[key]) for row in rows]))
    return summary
