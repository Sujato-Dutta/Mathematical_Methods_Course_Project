from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.noise import EPS, ensure_looks_tensor, gamma_noise_strength
from src.pde_baseline import gaussian_kernel, directional_differences


class DifferentiablePDE(nn.Module):
    def __init__(self, num_iterations: int = 20, delta_t: float = 0.12, alpha: float = 1.0, beta: float = 2.0, sigma: float = 1.0):
        super().__init__()
        self.num_iterations = num_iterations
        self.delta_t = delta_t
        self.sigma = sigma
        
        # Learnable parameters initialized at tuned defaults
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))

        kernel_size = 5 if self.sigma <= 1.0 else 7
        gaussian = gaussian_kernel(size=kernel_size, sigma=self.sigma)
        self.register_buffer('gaussian', gaussian)
        self.pad = kernel_size // 2

    def forward(self, image: torch.Tensor, looks: int | torch.Tensor) -> torch.Tensor:
        looks_tensor = ensure_looks_tensor(looks, batch_size=image.shape[0], device=image.device, dtype=image.dtype)
        looks_value = looks_tensor.view(-1, 1, 1, 1)

        u = image.clone()
        for _ in range(self.num_iterations):
            u_sigma = F.conv2d(F.pad(u, (self.pad, self.pad, self.pad, self.pad), mode='replicate'), self.gaussian)
            grad_n_sigma, grad_s_sigma, grad_w_sigma, grad_e_sigma = directional_differences(u_sigma)
            grad_mag = torch.sqrt(
                0.25 * (grad_n_sigma.pow(2) + grad_s_sigma.pow(2) + grad_w_sigma.pow(2) + grad_e_sigma.pow(2)) + EPS
            )
            max_gray = u_sigma.amax(dim=(2, 3), keepdim=True).clamp_min(EPS)
            
            # Use self.alpha and self.beta parameters
            diffusion_coeff = (u_sigma / max_gray).clamp_min(EPS).pow(self.alpha) / (1.0 + grad_mag.pow(self.beta))

            diff_n, diff_s, diff_w, diff_e = directional_differences(u)
            coeff_n = 0.5 * (diffusion_coeff + F.pad(diffusion_coeff[:, :, :-1, :], (0, 0, 1, 0), mode='replicate'))
            coeff_s = 0.5 * (diffusion_coeff + F.pad(diffusion_coeff[:, :, 1:, :], (0, 0, 0, 1), mode='replicate'))
            coeff_w = 0.5 * (diffusion_coeff + F.pad(diffusion_coeff[:, :, :, :-1], (1, 0, 0, 0), mode='replicate'))
            coeff_e = 0.5 * (diffusion_coeff + F.pad(diffusion_coeff[:, :, :, 1:], (0, 1, 0, 0), mode='replicate'))

            divergence = coeff_n * diff_n + coeff_s * diff_s + coeff_w * diff_w + coeff_e * diff_e
            u = (u + self.delta_t * divergence).clamp(EPS, 1.0)
        return u


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # Global average pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class EnhancedInfluenceFunction(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.se_block = SEBlock(in_channels, reduction=4)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = self.se_block(x)
        x = torch.tanh(self.conv2(x))
        return x


class LevelEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(16, 32)

    def forward(self, looks: torch.Tensor) -> torch.Tensor:
        x = self.fc1(looks)
        x = self.relu(x)
        x = self.fc2(x)
        return x.view(-1, 32, 1, 1)


class HybridPlusStage(nn.Module):
    def __init__(self):
        super().__init__()
        # Two parallel learnable filter banks: 16 of 3x3, and 16 of 5x5
        # Input to filters will be 2 channels: u_prev and noisy input f
        
        f3x3 = torch.randn(16, 2, 3, 3) / math.sqrt(3 * 3 * 2)
        self.filters_3x3 = nn.Parameter(f3x3)
        
        f5x5 = torch.randn(16, 2, 5, 5) / math.sqrt(5 * 5 * 2)
        self.filters_5x5 = nn.Parameter(f5x5)
        
        self.influence = EnhancedInfluenceFunction(in_channels=32)
        self.lambda_param = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(self, u_prev: torch.Tensor, noisy: torch.Tensor, intensity_weight: torch.Tensor, level_bias: torch.Tensor) -> torch.Tensor:
        # Skip connection: concatenate original noisy input f to current estimate (2-channel)
        stage_input = torch.cat([u_prev, noisy], dim=1)

        # Multiscale filter bank
        resp_3x3 = F.conv2d(F.pad(stage_input, (1, 1, 1, 1), mode='reflect'), self.filters_3x3)
        resp_5x5 = F.conv2d(F.pad(stage_input, (2, 2, 2, 2), mode='reflect'), self.filters_5x5)
        responses = torch.cat([resp_3x3, resp_5x5], dim=1)  # 32 channels

        # Add noise level embedding
        responses = responses + level_bias

        # Enhanced influence function (32 -> SE attention -> 1 channel)
        influenced = self.influence(responses)

        # Diffusion term
        diffusion = intensity_weight * influenced

        # Data fidelity (reaction) term
        fidelity_grad = (u_prev - noisy) / (u_prev.square() + EPS)
        reaction = self.lambda_param * intensity_weight * fidelity_grad

        # Stage update
        estimate = u_prev - diffusion - reaction
        return estimate.clamp(EPS, 1.0)


class HybridPlusModel(nn.Module):
    def __init__(self, num_stages: int = 10, alpha: float = 1.0, beta: float = 2.0):
        super().__init__()
        self.pde = DifferentiablePDE(alpha=alpha, beta=beta)
        self.level_embedder = LevelEmbedding()
        self.stages = nn.ModuleList([HybridPlusStage() for _ in range(num_stages)])
        self.register_buffer('gaussian', gaussian_kernel(size=5, sigma=1.0))

    def compute_u_sigma(self, estimate: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(estimate, (2, 2, 2, 2), mode='reflect'), self.gaussian)

    def forward(self, noisy: torch.Tensor, looks: int | torch.Tensor, upto_stage: int | None = None) -> torch.Tensor:
        # PDE Initialization
        estimate = self.pde(noisy, looks)
        
        # Level bias
        looks_tensor = ensure_looks_tensor(looks, batch_size=noisy.shape[0], device=noisy.device, dtype=noisy.dtype)
        level_val = gamma_noise_strength(looks_tensor, batch_size=noisy.shape[0], device=noisy.device, dtype=noisy.dtype)
        level_bias = self.level_embedder(level_val.view(-1, 1))

        # Stages
        active_stages = self.stages if upto_stage is None else self.stages[:upto_stage]
        for stage in active_stages:
            u_sigma = self.compute_u_sigma(estimate)
            intensity_weight = (u_sigma / u_sigma.amax(dim=(2, 3), keepdim=True).clamp_min(EPS)).clamp_min(EPS)
            estimate = stage(estimate, noisy, intensity_weight, level_bias)
            
        return estimate.clamp(EPS, 1.0)
