from __future__ import annotations

import torch


def add_gamma_noise(
    clean: torch.Tensor,
    looks: int,
) -> torch.Tensor:
    if looks <= 0:
        raise ValueError("looks must be positive")

    concentration = torch.full_like(clean, float(looks))
    rate = torch.full_like(clean, float(looks))
    gamma = torch.distributions.Gamma(concentration=concentration, rate=rate)
    noise = gamma.sample()
    noisy = clean * noise
    return noisy.clamp_(1e-4, 1.0)
