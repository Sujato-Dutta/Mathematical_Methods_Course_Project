from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.noise import EPS


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel.view(1, 1, size, size)


def directional_differences(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    north = F.pad(image[:, :, :-1, :], (0, 0, 1, 0), mode='replicate') - image
    south = F.pad(image[:, :, 1:, :], (0, 0, 0, 1), mode='replicate') - image
    west = F.pad(image[:, :, :, :-1], (1, 0, 0, 0), mode='replicate') - image
    east = F.pad(image[:, :, :, 1:], (0, 1, 0, 0), mode='replicate') - image
    return north, south, west, east


@dataclass
class PDEConfig:
    num_iterations: int = 20
    delta_t: float = 0.12
    alpha: float = 1.0
    beta: float = 2.0
    sigma: float = 1.0


def nonlinear_smooth_diffusion_denoise(image: torch.Tensor, looks: int, config: PDEConfig | None = None) -> torch.Tensor:
    if looks <= 0:
        raise ValueError('looks must be positive')
    config = config or PDEConfig()
    kernel_size = 5 if config.sigma <= 1.0 else 7
    gaussian = gaussian_kernel(size=kernel_size, sigma=config.sigma).to(device=image.device, dtype=image.dtype)
    pad = kernel_size // 2
    u = image.clone()

    for _ in range(config.num_iterations):
        u_sigma = F.conv2d(F.pad(u, (pad, pad, pad, pad), mode='replicate'), gaussian)
        grad_n_sigma, grad_s_sigma, grad_w_sigma, grad_e_sigma = directional_differences(u_sigma)
        grad_mag = torch.sqrt(
            0.25 * (grad_n_sigma.pow(2) + grad_s_sigma.pow(2) + grad_w_sigma.pow(2) + grad_e_sigma.pow(2)) + EPS
        )
        max_gray = u_sigma.amax(dim=(2, 3), keepdim=True).clamp_min(EPS)
        diffusion_coeff = (u_sigma / max_gray).clamp_min(EPS).pow(config.alpha) / (1.0 + grad_mag.pow(config.beta))

        diff_n, diff_s, diff_w, diff_e = directional_differences(u)
        coeff_n = 0.5 * (diffusion_coeff + F.pad(diffusion_coeff[:, :, :-1, :], (0, 0, 1, 0), mode='replicate'))
        coeff_s = 0.5 * (diffusion_coeff + F.pad(diffusion_coeff[:, :, 1:, :], (0, 0, 0, 1), mode='replicate'))
        coeff_w = 0.5 * (diffusion_coeff + F.pad(diffusion_coeff[:, :, :, :-1], (1, 0, 0, 0), mode='replicate'))
        coeff_e = 0.5 * (diffusion_coeff + F.pad(diffusion_coeff[:, :, :, 1:], (0, 1, 0, 0), mode='replicate'))

        divergence = coeff_n * diff_n + coeff_s * diff_s + coeff_w * diff_w + coeff_e * diff_e
        u = (u + config.delta_t * divergence).clamp(EPS, 1.0)
    return u
