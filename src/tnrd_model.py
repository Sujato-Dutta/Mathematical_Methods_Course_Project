from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.view(1, 1, size, size)


class InfluenceFunction(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, groups=num_channels, bias=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IdentityInfluence(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TNRDStage(nn.Module):
    def __init__(
        self,
        filters: torch.Tensor,
        use_nonlinearity: bool = True,
        epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("filters", filters.clone())
        flipped = torch.flip(filters, dims=(-1, -2))
        self.register_buffer("flipped_filters", flipped)
        self.influence = InfluenceFunction(filters.shape[0]) if use_nonlinearity else IdentityInfluence()
        self.lambda_param = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(
        self,
        u_prev: torch.Tensor,
        noisy: torch.Tensor,
        looks: int,
        u_sigma: torch.Tensor,
    ) -> torch.Tensor:
        # Formula mapping:
        # u_prev -> u_{t-1}
        # noisy -> f
        # looks  -> M (gamma noise parameter, L in the experiments)
        padded = F.pad(u_prev, (1, 1, 1, 1), mode="reflect")
        responses = F.conv2d(padded, self.filters)
        influenced = self.influence(responses)
        gamma_param = float(looks)
        weighted = (u_sigma / gamma_param) * influenced
        diffusion = F.conv_transpose2d(weighted, self.flipped_filters, padding=1)
        reaction = self.lambda_param * ((u_prev - noisy) / (u_prev.pow(2) + self.epsilon))
        return (u_prev - (diffusion + reaction)).clamp(1e-4, 1.0)


class TNRDModel(nn.Module):
    def __init__(
        self,
        num_filters: int = 8,
        kernel_size: int = 3,
        num_stages: int = 5,
        use_nonlinearity: bool = True,
        epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        if kernel_size != 3:
            raise ValueError("This implementation expects 3x3 filters.")

        self.register_buffer("gaussian", gaussian_kernel(size=5, sigma=1.0))
        self.stages = nn.ModuleList()
        for _ in range(num_stages):
            filters = torch.randn(num_filters, 1, kernel_size, kernel_size) / math.sqrt(kernel_size**2)
            self.stages.append(
                TNRDStage(filters=filters, use_nonlinearity=use_nonlinearity, epsilon=epsilon)
            )
        for stage in self.stages:
            stage.filters.requires_grad = False
            stage.flipped_filters.requires_grad = False

    def compute_u_sigma(self, u_prev: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(u_prev, (2, 2, 2, 2), mode="reflect"), self.gaussian)

    def forward(self, noisy: torch.Tensor, looks: int, upto_stage: int | None = None) -> torch.Tensor:
        u = noisy.clone()
        active_stages = self.stages if upto_stage is None else self.stages[:upto_stage]
        for stage in active_stages:
            u_sigma = self.compute_u_sigma(u)
            u = stage(u_prev=u, noisy=noisy, looks=looks, u_sigma=u_sigma)
        return u
