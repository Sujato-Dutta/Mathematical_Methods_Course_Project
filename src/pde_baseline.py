from __future__ import annotations

import torch
import torch.nn.functional as F

from src.tnrd_model import gaussian_kernel


def _directional_differences(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    north = F.pad(image[:, :, :-1, :], (0, 0, 1, 0), mode="replicate") - image
    south = F.pad(image[:, :, 1:, :], (0, 0, 0, 1), mode="replicate") - image
    west = F.pad(image[:, :, :, :-1], (1, 0, 0, 0), mode="replicate") - image
    east = F.pad(image[:, :, :, 1:], (0, 1, 0, 0), mode="replicate") - image
    return north, south, west, east


def nonlinear_smooth_diffusion_denoise(
    image: torch.Tensor,
    looks: int,
    num_iterations: int = 20,
    delta_t: float = 0.12,
    alpha: float = 1.0,
    beta: float = 2.0,
    sigma: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if looks <= 0:
        raise ValueError("looks must be positive")

    kernel_size = 5 if sigma <= 1.0 else 7
    gaussian = gaussian_kernel(size=kernel_size, sigma=sigma).to(device=image.device, dtype=image.dtype)
    u = image.clone()

    for _ in range(num_iterations):
        pad = kernel_size // 2
        u_sigma = F.conv2d(F.pad(u, (pad, pad, pad, pad), mode="replicate"), gaussian)
        grad_n_sigma, grad_s_sigma, grad_w_sigma, grad_e_sigma = _directional_differences(u_sigma)
        grad_mag = torch.sqrt(
            0.25
            * (
                grad_n_sigma.pow(2)
                + grad_s_sigma.pow(2)
                + grad_w_sigma.pow(2)
                + grad_e_sigma.pow(2)
            )
            + epsilon
        )
        max_gray = u_sigma.amax(dim=(2, 3), keepdim=True).clamp_min(epsilon)
        diffusion_coeff = (u_sigma / max_gray).clamp_min(epsilon).pow(alpha) / (1.0 + grad_mag.pow(beta))

        diff_n, diff_s, diff_w, diff_e = _directional_differences(u)
        coeff_n = 0.5 * (diffusion_coeff + F.pad(diffusion_coeff[:, :, :-1, :], (0, 0, 1, 0), mode="replicate"))
        coeff_s = 0.5 * (diffusion_coeff + F.pad(diffusion_coeff[:, :, 1:, :], (0, 0, 0, 1), mode="replicate"))
        coeff_w = 0.5 * (diffusion_coeff + F.pad(diffusion_coeff[:, :, :, :-1], (1, 0, 0, 0), mode="replicate"))
        coeff_e = 0.5 * (diffusion_coeff + F.pad(diffusion_coeff[:, :, :, 1:], (0, 1, 0, 0), mode="replicate"))

        divergence = coeff_n * diff_n + coeff_s * diff_s + coeff_w * diff_w + coeff_e * diff_e
        u = (u + delta_t * divergence).clamp(1e-4, 1.0)

    return u
