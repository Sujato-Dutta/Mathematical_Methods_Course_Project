from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_loss(estimate: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(estimate, clean)
