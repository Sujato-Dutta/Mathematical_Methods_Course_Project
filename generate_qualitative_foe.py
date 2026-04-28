"""Generate qualitative comparison figures for FoE test set.

Produces two figures (L=1 and L=10), each showing 3 test images across 7 columns:
Clean | Noisy | BM3D | PDE | TNRD | Hybrid++ (Ours) | DnCNN

Usage:
    python generate_qualitative_foe.py
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import polygamma

from src.benchmark_models import DnCNN
from src.dataset import build_loader, discover_images, split_dataset
from src.hybrid_plus_v2 import HybridPlusModelV2
from src.noise import LOOK_LEVELS, add_gamma_noise
from src.pde_baseline import PDEConfig, nonlinear_smooth_diffusion_denoise
from src.tnrd_model import StagewiseTNRD
from src.train import load_checkpoint
from src.utils import get_device, set_seed, tensor_to_numpy


def bm3d_denoise_gamma(noisy_np: np.ndarray, looks: int) -> np.ndarray:
    import bm3d
    eps = 1e-10
    log_noisy = np.log(np.clip(noisy_np, eps, None))
    sigma_log = float(np.sqrt(polygamma(1, looks)))
    log_denoised = bm3d.bm3d(log_noisy, sigma_psd=sigma_log)
    return np.clip(np.exp(log_denoised), 0.0, 1.0)


@torch.no_grad()
def main() -> None:
    set_seed(42)
    device = get_device()
    print(f'Device: {device}')

    figures_dir = Path('./figures')
    figures_dir.mkdir(exist_ok=True)
    ckpt_dir = Path('./results/results/checkpoints')

    # --- Load all models ---
    pde_config = PDEConfig(alpha=1.0, beta=2.0, delta_t=0.12, num_iterations=20, sigma=1.0)

    dncnn_models = {}
    for looks in LOOK_LEVELS:
        m = DnCNN().to(device)
        load_checkpoint(ckpt_dir / f'dncnn_L{looks}.pth', m, device)
        m.eval()
        dncnn_models[looks] = m

    tnrd_models = {}
    for looks in LOOK_LEVELS:
        m = StagewiseTNRD(num_filters=8, num_stages=7, use_nonlinearity=True,
                          use_skip=True, learnable_filters=False, use_level_embedding=False).to(device)
        load_checkpoint(ckpt_dir / f'tnrd_L{looks}.pth', m, device)
        m.eval()
        tnrd_models[looks] = m

    hybrid = HybridPlusModelV2(num_stages=3, alpha=1.0, beta=2.0).to(device)
    load_checkpoint(Path('./results_hybrid_plus_v2_finetuned/checkpoints/finetuned.pt'), hybrid, device)
    hybrid.eval()

    print('All models loaded.')

    # --- Load FoE test set (same seed-42 split) ---
    foe_paths = discover_images('./data/FoETrainingSets180')
    foe_split = split_dataset(foe_paths, seed=42)
    test_loader = build_loader(foe_split.test, batch_size=1, shuffle=False, augment=False)

    # Collect first 3 test images
    test_images = []
    for i, batch in enumerate(test_loader):
        if i >= 3:
            break
        test_images.append({'clean': batch['image'].to(device), 'name': batch['name'][0]})

    # --- Generate figures for each noise level ---
    for looks in LOOK_LEVELS:
        fig, axes = plt.subplots(3, 7, figsize=(21, 9))
        col_titles = ['Clean', 'Noisy', 'BM3D', 'PDE', 'TNRD', 'Hybrid++ (Ours)', 'DnCNN']

        for row, img_data in enumerate(test_images):
            clean = img_data['clean']
            noisy = add_gamma_noise(clean, looks)

            # BM3D (numpy)
            noisy_np = noisy.cpu().squeeze().numpy()
            bm3d_out = bm3d_denoise_gamma(noisy_np, looks)

            # PDE
            pde_out = nonlinear_smooth_diffusion_denoise(noisy, looks=looks, config=pde_config)

            # TNRD
            tnrd_out = tnrd_models[looks](noisy, looks)

            # Hybrid++
            hybrid_out = hybrid(noisy, looks)

            # DnCNN
            dncnn_out = dncnn_models[looks](noisy, looks)

            # Plot all 7 columns
            results = [
                tensor_to_numpy(clean),
                tensor_to_numpy(noisy),
                bm3d_out,
                tensor_to_numpy(pde_out),
                tensor_to_numpy(tnrd_out),
                tensor_to_numpy(hybrid_out),
                tensor_to_numpy(dncnn_out),
            ]

            for col, img in enumerate(results):
                axes[row, col].imshow(img, cmap='gray', vmin=0.0, vmax=1.0)
                axes[row, col].axis('off')
                if row == 0:
                    axes[row, col].set_title(col_titles[col], fontsize=14, fontweight='bold')

        plt.suptitle(f'Qualitative Comparison — FoE Test Set (L={looks})', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = figures_dir / f'fig_qualitative_foe_L{looks}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out_path}')

    print('Done!')


if __name__ == '__main__':
    main()
