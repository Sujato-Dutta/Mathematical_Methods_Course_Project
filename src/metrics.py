from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch

from src.noise import LOOK_LEVELS
from src.utils import tensor_to_numpy

MODEL_ORDER = ('Noisy', 'PDE', 'DnCNN', 'FFDNet', 'TNRD', 'Hybrid')


def compute_psnr(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    return float(peak_signal_noise_ratio(tensor_to_numpy(reference), tensor_to_numpy(estimate), data_range=1.0))


def compute_ssim(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    return float(structural_similarity(tensor_to_numpy(reference), tensor_to_numpy(estimate), data_range=1.0))


def aggregate_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row['looks']), str(row['model']))].append(row)

    summary_rows: list[dict[str, object]] = []
    for looks in LOOK_LEVELS:
        for model_name in MODEL_ORDER:
            subset = grouped.get((looks, model_name), [])
            if not subset:
                continue
            psnr_values = np.array([float(row['psnr']) for row in subset], dtype=np.float32)
            ssim_values = np.array([float(row['ssim']) for row in subset], dtype=np.float32)
            summary_rows.append(
                {
                    'looks': looks,
                    'model': model_name,
                    'psnr_mean': float(psnr_values.mean()),
                    'psnr_std': float(psnr_values.std()),
                    'ssim_mean': float(ssim_values.mean()),
                    'ssim_std': float(ssim_values.std()),
                }
            )
    return summary_rows
