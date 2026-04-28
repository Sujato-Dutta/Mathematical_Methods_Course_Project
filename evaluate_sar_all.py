"""Denoise real SAR images with PDE, DnCNN, Original TNRD, and Hybrid++ V2 (finetuned).

SAR images are already noisy — no synthetic noise is added.
No PSNR/SSIM (no clean ground truth exists). Output is purely qualitative.

Usage:
    python evaluate_sar_all.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.benchmark_models import DnCNN
from src.hybrid_plus_v2 import HybridPlusModelV2
from src.noise import LOOK_LEVELS
from src.pde_baseline import PDEConfig, nonlinear_smooth_diffusion_denoise
from src.tnrd_model import StagewiseTNRD
from src.train import load_checkpoint
from src.utils import ensure_dir, get_device, save_image, save_json, set_seed


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


class SARImageDataset(Dataset):
    def __init__(self, sar_path: str | Path) -> None:
        self.root = Path(sar_path)
        self.paths = sorted(
            p for p in self.root.rglob('*')
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise FileNotFoundError(f'No SAR images found under {self.root}')

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Failed to read image: {path}')
        tensor = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
        return {'image': tensor, 'name': path.stem}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description='Denoise real SAR images with 4 models')
    parser.add_argument('--sar_path', default='./data/SAR')
    parser.add_argument('--results_dir', default='./results_SAR')
    parser.add_argument('--baseline_checkpoints', default='./results/results/checkpoints')
    parser.add_argument('--hybrid_checkpoint', default='./results_hybrid_plus_v2_finetuned/checkpoints/finetuned.pt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f'Using device: {device}')

    results_root = ensure_dir(Path(args.results_dir))
    ckpt_dir = Path(args.baseline_checkpoints)

    # Load SAR dataset
    dataset = SARImageDataset(args.sar_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f'SAR images: {len(dataset)}')

    # PDE (classical, no checkpoint)
    pde_config = PDEConfig(alpha=1.0, beta=2.0, delta_t=0.12, num_iterations=20, sigma=1.0)

    # DnCNN (separate checkpoint per looks)
    dncnn_models: dict[int, torch.nn.Module] = {}
    for looks in LOOK_LEVELS:
        model = DnCNN().to(device)
        load_checkpoint(ckpt_dir / f'dncnn_L{looks}.pth', model, device)
        model.eval()
        dncnn_models[looks] = model
    print('Loaded DnCNN checkpoints')

    # Original TNRD (separate checkpoint per looks)
    tnrd_models: dict[int, torch.nn.Module] = {}
    for looks in LOOK_LEVELS:
        model = StagewiseTNRD(
            num_filters=8, num_stages=7,
            use_nonlinearity=True, use_skip=True,
            learnable_filters=False, use_level_embedding=False,
        ).to(device)
        load_checkpoint(ckpt_dir / f'tnrd_L{looks}.pth', model, device)
        model.eval()
        tnrd_models[looks] = model
    print('Loaded TNRD checkpoints')

    # Hybrid++ V2 finetuned (single unified model)
    hybrid_v2 = HybridPlusModelV2(num_stages=3, alpha=1.0, beta=2.0).to(device)
    load_checkpoint(Path(args.hybrid_checkpoint), hybrid_v2, device)
    hybrid_v2.eval()
    print(f'Loaded Hybrid++ V2 finetuned')

    # --- Denoise each SAR image (already noisy, no synthetic noise added) ---
    # SAR speckle is typically modeled as L=1 (single-look)
    # We also run L=10 models to show generalization

    summary_info = []

    for looks in LOOK_LEVELS:
        image_dir = ensure_dir(results_root / f'denoised_L{looks}')
        print(f'\n--- Denoising SAR images assuming L={looks} ---')

        for batch in tqdm(loader, desc=f'SAR L={looks}'):
            noisy = batch['image'].to(device)
            name = str(batch['name'][0])

            # Save original (noisy) SAR image
            save_image(image_dir / f'{name}_original.png', noisy)

            # Denoise with each model
            pde_out = nonlinear_smooth_diffusion_denoise(noisy, looks=looks, config=pde_config)
            tnrd_out = tnrd_models[looks](noisy, looks)
            hybrid_out = hybrid_v2(noisy, looks)
            dncnn_out = dncnn_models[looks](noisy, looks)

            save_image(image_dir / f'{name}_PDE.png', pde_out)
            save_image(image_dir / f'{name}_TNRD.png', tnrd_out)
            save_image(image_dir / f'{name}_HybridPP.png', hybrid_out)
            save_image(image_dir / f'{name}_DnCNN.png', dncnn_out)

            summary_info.append({
                'image': name,
                'looks': looks,
                'models_applied': ['PDE', 'TNRD', 'Hybrid++', 'DnCNN'],
            })

    save_json(results_root / 'summary.json', {
        'dataset': 'SAR (real speckle)',
        'num_images': len(dataset),
        'note': 'No PSNR/SSIM — no clean ground truth exists for real SAR images. Results are qualitative only.',
        'denoised_images': summary_info,
    })
    print(f'\nAll SAR denoised images saved to {results_root.resolve()}')
    print('Note: PSNR/SSIM are not computed — real SAR images have no clean ground truth.')


if __name__ == '__main__':
    main()
