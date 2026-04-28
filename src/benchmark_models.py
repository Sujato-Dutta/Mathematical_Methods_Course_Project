from __future__ import annotations

import torch
import torch.nn as nn

from src.noise import EPS, gamma_noise_strength


class DnCNN(nn.Module):
    def __init__(self, depth: int = 17, num_features: int = 64) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(1, num_features, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(num_features, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, noisy: torch.Tensor, looks: int | torch.Tensor | None = None) -> torch.Tensor:
        residual = self.net(noisy)
        return (noisy - residual).clamp(EPS, 1.0)


class FFDNetGray(nn.Module):
    def __init__(self, num_features: int = 64, num_layers: int = 15) -> None:
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.shuffle = nn.PixelShuffle(2)
        layers: list[nn.Module] = [nn.Conv2d(5, num_features, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(inplace=True)])
        layers.append(nn.Conv2d(num_features, 4, kernel_size=3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, noisy: torch.Tensor, looks: int | torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = noisy.shape
        pad_h = height % 2
        pad_w = width % 2
        if pad_h or pad_w:
            noisy = nn.functional.pad(noisy, (0, pad_w, 0, pad_h), mode='reflect')
        downsampled = self.unshuffle(noisy)
        sigma = gamma_noise_strength(looks, batch_size=batch_size, device=noisy.device, dtype=noisy.dtype)
        sigma_map = sigma.expand(batch_size, 1, downsampled.shape[-2], downsampled.shape[-1])
        residual = self.body(torch.cat([downsampled, sigma_map], dim=1))
        estimate = (noisy - self.shuffle(residual)).clamp(EPS, 1.0)
        if pad_h or pad_w:
            estimate = estimate[:, :, :height, :width]
        return estimate
