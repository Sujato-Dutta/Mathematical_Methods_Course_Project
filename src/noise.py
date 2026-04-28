from __future__ import annotations

import torch

EPS = 1e-6
LOOK_LEVELS = (1, 10)


def ensure_looks_tensor(
    looks: int | float | torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(looks, torch.Tensor):
        tensor = looks.to(device=device, dtype=dtype)
        if tensor.ndim == 0:
            tensor = tensor.view(1).repeat(batch_size)
        return tensor.view(batch_size, 1, 1, 1)
    return torch.full((batch_size, 1, 1, 1), float(looks), device=device, dtype=dtype)


def gamma_noise_strength(looks: int | float | torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    looks_tensor = ensure_looks_tensor(looks, batch_size=batch_size, device=device, dtype=dtype)
    return looks_tensor.rsqrt()


def add_gamma_noise(clean: torch.Tensor, looks: int | float | torch.Tensor) -> torch.Tensor:
    looks_tensor = ensure_looks_tensor(looks, batch_size=clean.shape[0], device=clean.device, dtype=clean.dtype)
    concentration = looks_tensor.expand_as(clean)
    rate = concentration
    gamma = torch.distributions.Gamma(concentration=concentration, rate=rate)
    noisy = clean * gamma.sample()
    return noisy.clamp(EPS, 1.0)
