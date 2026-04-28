"""Evaluate BM3D on the FoE test set (40 images) using homomorphic approach for multiplicative gamma noise.

Usage:
    pip install bm3d          # one-time dependency
    python evaluate_bm3d.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.special import polygamma
from tqdm import tqdm

from src.dataset import build_loader, discover_images, split_dataset
from src.metrics import compute_psnr, compute_ssim
from src.noise import LOOK_LEVELS, add_gamma_noise
from src.utils import ensure_dir, get_device, save_json, set_seed


def bm3d_denoise_gamma(noisy_np: np.ndarray, looks: int) -> np.ndarray:
    """Homomorphic BM3D: log → BM3D (additive Gaussian) → exp.

    For Gamma(L, 1/L) noise, the log-domain noise variance is the
    trigamma function ψ₁(L).  We pass σ = sqrt(ψ₁(L)) to BM3D.
    """
    import bm3d

    eps = 1e-10
    log_noisy = np.log(np.clip(noisy_np, eps, None))

    # Trigamma gives the variance of log(Gamma(L,1/L))
    sigma_log = float(np.sqrt(polygamma(1, looks)))

    log_denoised = bm3d.bm3d(log_noisy, sigma_psd=sigma_log)
    denoised = np.exp(log_denoised)
    return np.clip(denoised, 0.0, 1.0)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate BM3D on FoE test set')
    parser.add_argument('--foe_path', default='./data/FoETrainingSets180')
    parser.add_argument('--results_dir', default='./results/BM3D')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f'Using device: {device}')

    results_root = ensure_dir(Path(args.results_dir))

    # Same seed-42 split as all other models
    foe_paths = discover_images(args.foe_path)
    foe_split = split_dataset(foe_paths, seed=args.seed)
    test_loader = build_loader(foe_split.test, batch_size=1, shuffle=False, augment=False)
    print(f'FoE test set: {len(foe_split.test)} images')

    all_summary = []

    for looks in LOOK_LEVELS:
        rows = []
        for batch in tqdm(test_loader, desc=f'BM3D L={looks}'):
            clean = batch['image'].to(device)
            name = str(batch['name'][0])
            noisy = add_gamma_noise(clean, looks)

            # BM3D operates on numpy (H, W)
            noisy_np = noisy.detach().cpu().squeeze().numpy()
            denoised_np = bm3d_denoise_gamma(noisy_np, looks)

            # Back to tensor for metric computation
            estimate = torch.from_numpy(denoised_np).float().unsqueeze(0).unsqueeze(0).to(device)

            rows.append({
                'image': name,
                'psnr': compute_psnr(clean, estimate),
                'ssim': compute_ssim(clean, estimate),
            })

        # Per-image CSV
        csv_path = results_root / f'metrics_L{looks}.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'psnr', 'ssim'])
            writer.writeheader()
            writer.writerows(rows)

        psnr_arr = np.array([r['psnr'] for r in rows])
        ssim_arr = np.array([r['ssim'] for r in rows])

        summary = {
            'looks': looks,
            'psnr_mean': float(psnr_arr.mean()),
            'psnr_std': float(psnr_arr.std()),
            'ssim_mean': float(ssim_arr.mean()),
            'ssim_std': float(ssim_arr.std()),
        }
        all_summary.append(summary)
        print(f'L={looks}:  PSNR = {summary["psnr_mean"]:.4f} ± {summary["psnr_std"]:.4f}  |  '
              f'SSIM = {summary["ssim_mean"]:.4f} ± {summary["ssim_std"]:.4f}')

    save_json(results_root / 'summary.json', {
        'model': 'BM3D_homomorphic',
        'seed': args.seed,
        'test_evaluation': all_summary,
    })
    print(f'\nBM3D results saved to {results_root.resolve()}')


if __name__ == '__main__':
    main()
