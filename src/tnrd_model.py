from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.noise import EPS, ensure_looks_tensor, gamma_noise_strength
from src.pde_baseline import PDEConfig, gaussian_kernel, nonlinear_smooth_diffusion_denoise


class InfluenceFunction(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(x))


class IdentityInfluence(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ReactionDiffusionStage(nn.Module):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int = 3,
        use_nonlinearity: bool = True,
        use_skip: bool = True,
        learnable_filters: bool = False,
        use_level_embedding: bool = False,
    ) -> None:
        super().__init__()
        filters = torch.randn(num_filters, 1, kernel_size, kernel_size) / math.sqrt(kernel_size * kernel_size)
        if learnable_filters:
            self.filters = nn.Parameter(filters)
        else:
            self.register_buffer('filters', filters)
        self.influence = InfluenceFunction(num_filters) if use_nonlinearity else IdentityInfluence()
        self.lambda_param = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.skip_gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32)) if use_skip else None
        self.level_projection = nn.Linear(1, num_filters) if use_level_embedding else None

    def forward(
        self,
        u_prev: torch.Tensor,
        noisy: torch.Tensor,
        intensity_weight: torch.Tensor,
        looks: int | torch.Tensor,
    ) -> torch.Tensor:
        filters = self.filters
        responses = F.conv2d(F.pad(u_prev, (1, 1, 1, 1), mode='reflect'), filters)
        if self.level_projection is not None:
            level_signal = gamma_noise_strength(looks, batch_size=u_prev.shape[0], device=u_prev.device, dtype=u_prev.dtype)
            level_bias = self.level_projection(level_signal.view(u_prev.shape[0], 1)).view(u_prev.shape[0], -1, 1, 1)
            responses = responses + level_bias
        influenced = self.influence(responses)
        flipped = torch.flip(filters, dims=(-1, -2))
        diffusion = F.conv_transpose2d(intensity_weight * influenced, flipped, padding=1)
        fidelity_grad = (u_prev - noisy) / (u_prev.square() + EPS)
        reaction = self.lambda_param * intensity_weight * fidelity_grad
        estimate = u_prev - diffusion - reaction
        if self.skip_gate is not None:
            gate = torch.sigmoid(self.skip_gate)
            estimate = estimate + gate * (noisy - estimate)
        return estimate.clamp(EPS, 1.0)


class StagewiseTNRD(nn.Module):
    def __init__(
        self,
        num_filters: int = 8,
        num_stages: int = 7,
        use_nonlinearity: bool = True,
        use_skip: bool = True,
        learnable_filters: bool = False,
        use_level_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer('gaussian', gaussian_kernel(size=5, sigma=1.0))
        self.stages = nn.ModuleList(
            [
                ReactionDiffusionStage(
                    num_filters=num_filters,
                    kernel_size=3,
                    use_nonlinearity=use_nonlinearity,
                    use_skip=use_skip,
                    learnable_filters=learnable_filters,
                    use_level_embedding=use_level_embedding,
                )
                for _ in range(num_stages)
            ]
        )

    def initial_estimate(self, noisy: torch.Tensor, looks: int | torch.Tensor) -> torch.Tensor:
        del looks
        return noisy

    def compute_u_sigma(self, estimate: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(estimate, (2, 2, 2, 2), mode='reflect'), self.gaussian)

    def forward(self, noisy: torch.Tensor, looks: int | torch.Tensor, upto_stage: int | None = None) -> torch.Tensor:
        estimate = self.initial_estimate(noisy, looks)
        active_stages = self.stages if upto_stage is None else self.stages[:upto_stage]
        for stage in active_stages:
            u_sigma = self.compute_u_sigma(estimate)
            intensity_weight = (u_sigma / u_sigma.amax(dim=(2, 3), keepdim=True).clamp_min(EPS)).clamp_min(EPS)
            estimate = stage(estimate, noisy, intensity_weight, looks)
        return estimate.clamp(EPS, 1.0)


class HybridStagewiseTNRD(StagewiseTNRD):
    def __init__(self, pde_config: PDEConfig) -> None:
        super().__init__(
            num_filters=8,
            num_stages=6,
            use_nonlinearity=True,
            use_skip=True,
            learnable_filters=True,
            use_level_embedding=True,
        )
        self.pde_config = pde_config

    def initial_estimate(self, noisy: torch.Tensor, looks: int | torch.Tensor) -> torch.Tensor:
        looks_tensor = ensure_looks_tensor(looks, batch_size=noisy.shape[0], device=noisy.device, dtype=noisy.dtype)
        looks_value = int(float(looks_tensor[0, 0, 0, 0].item()))
        return nonlinear_smooth_diffusion_denoise(noisy, looks=looks_value, config=self.pde_config)
