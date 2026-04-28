from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import compute_psnr, compute_ssim
from src.noise import LOOK_LEVELS, add_gamma_noise
from src.tnrd_model import StagewiseTNRD
from src.train import TrainingConfig, train_stagewise_model


def build_ablation_model(variant_name: str) -> StagewiseTNRD:
    variant_specs = {
        'Full TNRD': {'num_filters': 8, 'num_stages': 7, 'use_nonlinearity': True, 'use_skip': True},
        'No skip connections': {'num_filters': 8, 'num_stages': 7, 'use_nonlinearity': True, 'use_skip': False},
        'No nonlinear influence': {'num_filters': 8, 'num_stages': 7, 'use_nonlinearity': False, 'use_skip': True},
        'Reduced filters': {'num_filters': 4, 'num_stages': 7, 'use_nonlinearity': True, 'use_skip': True},
        'Reduced stages': {'num_filters': 8, 'num_stages': 3, 'use_nonlinearity': True, 'use_skip': True},
    }
    spec = variant_specs[variant_name]
    model = StagewiseTNRD(
        num_filters=int(spec['num_filters']),
        num_stages=int(spec['num_stages']),
        use_nonlinearity=bool(spec['use_nonlinearity']),
        use_skip=bool(spec['use_skip']),
        learnable_filters=False,
        use_level_embedding=False,
    )
    return model


def run_ablation_suite(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    checkpoints_dir: Path,
    config: TrainingConfig,
    trained_tnrd: dict[int, StagewiseTNRD],
    results_root: Path,
) -> list[dict[str, object]]:
    variant_names = (
        'Full TNRD',
        'No skip connections',
        'No nonlinear influence',
        'Reduced filters',
        'Reduced stages',
    )
    rows: list[dict[str, object]] = []
    for looks in LOOK_LEVELS:
        for variant_name in variant_names:
            if variant_name == 'Full TNRD':
                model = trained_tnrd[looks]
                model.eval()
            else:
                model = build_ablation_model(variant_name)
                model = model.to(device)
                checkpoint_name = f"ablation_{variant_name.lower().replace(' ', '_').replace('-', '_')}_L{looks}.pth"
                train_stagewise_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    looks=looks,
                    model_name=f'{variant_name} L={looks}',
                    checkpoint_path=checkpoints_dir / checkpoint_name,
                    config=config,
                )

            metric_rows = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f'Ablation {variant_name} L={looks}', leave=False):
                    clean = batch['image'].to(device)
                    noisy = add_gamma_noise(clean, looks)
                    estimate = model(noisy, looks)
                    metric_rows.append({'psnr': compute_psnr(clean, estimate), 'ssim': compute_ssim(clean, estimate)})
            psnr_values = np.array([row['psnr'] for row in metric_rows], dtype=np.float32)
            ssim_values = np.array([row['ssim'] for row in metric_rows], dtype=np.float32)
            rows.append(
                {
                    'looks': looks,
                    'variant': variant_name,
                    'psnr_mean': float(psnr_values.mean()),
                    'psnr_std': float(psnr_values.std()),
                    'ssim_mean': float(ssim_values.mean()),
                    'ssim_std': float(ssim_values.std()),
                }
            )
    return rows
