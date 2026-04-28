from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import MODEL_ORDER, aggregate_rows, compute_psnr, compute_ssim
from src.noise import LOOK_LEVELS, add_gamma_noise
from src.pde_baseline import PDEConfig, nonlinear_smooth_diffusion_denoise
from src.utils import ensure_dir, save_image, tensor_to_numpy


@torch.no_grad()
def evaluate_foe_dataset(
    loader: DataLoader,
    device: torch.device,
    pde_config: PDEConfig,
    dncnn_models: dict[int, torch.nn.Module],
    ffdnet_model: torch.nn.Module,
    tnrd_models: dict[int, torch.nn.Module],
    hybrid_models: dict[int, torch.nn.Module],
    output_root: str | Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output_root = Path(output_root)
    figures_dir = ensure_dir(output_root / 'figures')
    denoised_dir = ensure_dir(output_root / 'denoised_images')

    metric_rows: list[dict[str, object]] = []
    for looks in LOOK_LEVELS:
        qualitative_examples: list[dict[str, torch.Tensor | str]] = []
        image_dir = ensure_dir(denoised_dir / f'L{looks}')
        print(f'Evaluating FoE at L={looks}')
        for batch in tqdm(loader, desc=f'Evaluating FoE L={looks}', leave=False):
            clean = batch['image'].to(device)
            name = str(batch['name'][0])
            noisy = add_gamma_noise(clean, looks)
            pde = nonlinear_smooth_diffusion_denoise(noisy, looks=looks, config=pde_config)
            dncnn = dncnn_models[looks](noisy, looks)
            ffdnet = ffdnet_model(noisy, looks)
            tnrd = tnrd_models[looks](noisy, looks)
            hybrid = hybrid_models[looks](noisy, looks)

            predictions = {
                'Noisy': noisy,
                'PDE': pde,
                'DnCNN': dncnn,
                'FFDNet': ffdnet,
                'TNRD': tnrd,
                'Hybrid': hybrid,
            }
            for model_name, estimate in predictions.items():
                metric_rows.append(
                    {
                        'dataset': 'foe',
                        'image': name,
                        'looks': looks,
                        'model': model_name,
                        'psnr': compute_psnr(clean, estimate),
                        'ssim': compute_ssim(clean, estimate),
                    }
                )
                save_image(image_dir / f'{name}_{model_name.lower()}.png', estimate)

            if len(qualitative_examples) < 3:
                qualitative_examples.append(
                    {
                        'name': name,
                        'clean': clean.cpu(),
                        'noisy': noisy.cpu(),
                        'dncnn': dncnn.cpu(),
                        'ffdnet': ffdnet.cpu(),
                        'pde': pde.cpu(),
                        'tnrd': tnrd.cpu(),
                        'hybrid': hybrid.cpu(),
                    }
                )

        if qualitative_examples:
            save_qualitative_figure(qualitative_examples, figures_dir / f'qualitative_L{looks}_foe.png')

    summary_rows = aggregate_rows(metric_rows)
    write_csv(output_root / 'results_table_foe.csv', summary_rows, ('looks', 'model', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std'))
    plot_psnr_comparison(summary_rows, output_root / 'figures' / 'psnr_comparison_foe.png', title='PSNR Comparison on FoE')
    return metric_rows, summary_rows


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_psnr_comparison(summary_rows: Sequence[dict[str, object]], output_path: Path, title: str) -> None:
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    color_map = {
        'Noisy': '#9e9e9e',
        'PDE': '#ff9800',
        'DnCNN': '#2196f3',
        'FFDNet': '#4caf50',
        'TNRD': '#795548',
        'Hybrid': '#d32f2f',
    }
    for looks in LOOK_LEVELS:
        for model_name in MODEL_ORDER:
            for row in summary_rows:
                if int(row['looks']) == looks and str(row['model']) == model_name:
                    labels.append(f'{model_name}\nL={looks}')
                    values.append(float(row['psnr_mean']))
                    colors.append(color_map[model_name])
                    break

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_qualitative_figure(examples: Sequence[dict[str, torch.Tensor | str]], output_path: Path) -> None:
    titles = ('Clean', 'Noisy', 'DnCNN', 'FFDNet', 'PDE', 'TNRD', 'Hybrid')
    fig, axes = plt.subplots(len(examples), len(titles), figsize=(14, 3 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, example in enumerate(examples):
        for col_index, key in enumerate(('clean', 'noisy', 'dncnn', 'ffdnet', 'pde', 'tnrd', 'hybrid')):
            axes[row_index, col_index].imshow(tensor_to_numpy(example[key]), cmap='gray', vmin=0.0, vmax=1.0)
            axes[row_index, col_index].axis('off')
            if row_index == 0:
                axes[row_index, col_index].set_title(titles[col_index])
        axes[row_index, 0].set_ylabel(str(example['name']), rotation=90, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def write_summary_text(results_root: Path, summary_rows: Sequence[dict[str, object]], pde_config: PDEConfig) -> None:
    lines = [
        'Gamma Denoising Experiment Summary',
        '',
        f'PDE validation-selected parameters: alpha={pde_config.alpha:.2f}, beta={pde_config.beta:.2f}, '
        f'delta_t={pde_config.delta_t:.2f}, iterations={pde_config.num_iterations}, sigma={pde_config.sigma:.2f}',
        '',
        'FOE',
    ]
    for looks in LOOK_LEVELS:
        lines.append(f'L={looks}')
        for row in summary_rows:
            if int(row['looks']) != looks:
                continue
            lines.append(
                f"{row['model']}: PSNR {float(row['psnr_mean']):.4f} +/- {float(row['psnr_std']):.4f}, "
                f"SSIM {float(row['ssim_mean']):.4f} +/- {float(row['ssim_std']):.4f}"
            )
        lines.append('')
    (results_root / 'summary.txt').write_text('\n'.join(lines).strip() + '\n', encoding='utf-8')
