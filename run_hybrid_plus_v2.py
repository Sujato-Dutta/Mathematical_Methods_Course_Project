from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import build_loader, discover_images, split_dataset
from src.hybrid_plus_v2 import HybridPlusModelV2
from src.losses import compute_loss
from src.metrics import compute_psnr, compute_ssim
from src.noise import LOOK_LEVELS, add_gamma_noise
from src.train import TrainingConfig, evaluate_validation_loss, load_checkpoint, save_checkpoint
from src.utils import ensure_dir, get_device, save_json, set_seed


# ---------------------------------------------------------------------------
# Stage-wise training (identical config to original hybrid_plus runner)
# ---------------------------------------------------------------------------

def train_hybrid_plus_v2(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    looks_options: list[int],
    checkpoint_path: Path,
    config: TrainingConfig,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, model, device)
        return {'best_val_loss': checkpoint.get('best_val_loss'), 'resumed_from_checkpoint': True}

    final_best_val = float('inf')
    stage_histories = []

    for stage_index in range(len(model.stages)):
        # Freeze previous stages
        for prior_stage in model.stages[:stage_index]:
            for parameter in prior_stage.parameters():
                parameter.requires_grad = False

        # Unfreeze current stage
        for parameter in model.stages[stage_index].parameters():
            parameter.requires_grad = True

        # PDE and level embedder always learnable
        for parameter in model.pde.parameters():
            parameter.requires_grad = True
        for parameter in model.level_embedder.parameters():
            parameter.requires_grad = True

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=config.scheduler_patience, factor=config.scheduler_factor,
        )

        best_val = float('inf')
        best_state = None
        patience = 0
        stage_history = {'train_loss': [], 'val_loss': []}

        epoch_bar = tqdm(range(config.epochs), desc=f'V2 Stage={stage_index + 1}/{len(model.stages)}', leave=False)
        for _ in epoch_bar:
            model.train()
            total_train = 0.0
            total_count = 0

            for batch in train_loader:
                clean = batch['image'].to(device)
                for looks in looks_options:
                    noisy = add_gamma_noise(clean, looks)
                    estimate = model(noisy, looks, upto_stage=stage_index + 1)
                    loss = compute_loss(estimate, clean)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
                    )
                    optimizer.step()

                    total_train += float(loss.item())
                    total_count += 1

            mean_train = total_train / max(1, total_count)
            mean_val = evaluate_validation_loss(model, val_loader, device, looks_options, upto_stage=stage_index + 1)

            scheduler.step(mean_val)
            stage_history['train_loss'].append(mean_train)
            stage_history['val_loss'].append(mean_val)
            epoch_bar.set_postfix(train=f'{mean_train:.4f}', val=f'{mean_val:.4f}')

            if mean_val < best_val:
                best_val = mean_val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= config.early_stopping_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        stage_histories.append({'stage': stage_index + 1, 'history': stage_history, 'best_val_loss': best_val})
        if stage_index == len(model.stages) - 1:
            final_best_val = best_val

    save_checkpoint(
        checkpoint_path, model,
        {'best_val_loss': final_best_val, 'looks_options': list(looks_options), 'stage_histories': stage_histories},
    )
    return {'best_val_loss': final_best_val, 'stage_histories': stage_histories}


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_test_set(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> list[dict]:
    model.eval()
    all_summary = []

    for looks in LOOK_LEVELS:
        rows = []
        for batch in tqdm(test_loader, desc=f'Testing L={looks}', leave=False):
            clean = batch['image'].to(device)
            name = str(batch['name'][0])
            noisy = add_gamma_noise(clean, looks)
            estimate = model(noisy, looks)

            rows.append({
                'image': name,
                'psnr': compute_psnr(clean, estimate),
                'ssim': compute_ssim(clean, estimate),
            })

        # Per-image CSV
        csv_path = output_dir / f'metrics_L{looks}.csv'
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

    return all_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train & Evaluate Hybrid++ V2')
    parser.add_argument('--foe_path', default='./data/FoETrainingSets180')
    parser.add_argument('--results_dir', default='./results_hybrid_plus_v2')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f'Using device: {device}')

    results_root = ensure_dir(Path(args.results_dir))
    checkpoints_dir = ensure_dir(results_root / 'checkpoints')

    # Dataset (seed 42 split)
    foe_paths = discover_images(args.foe_path)
    foe_split = split_dataset(foe_paths, seed=args.seed)
    print(f'FoE split | train={len(foe_split.train)}  val={len(foe_split.val)}  test={len(foe_split.test)}')

    train_config = TrainingConfig()  # 100 epochs, lr=1e-3, batch=4, patience=10
    train_loader = build_loader(foe_split.train, batch_size=train_config.batch_size, shuffle=True, augment=True)
    val_loader = build_loader(foe_split.val, batch_size=1, shuffle=False, augment=False)
    test_loader = build_loader(foe_split.test, batch_size=1, shuffle=False, augment=False)

    # Model: 7 stages, 64-channel filter banks
    model = HybridPlusModelV2(num_stages=7, alpha=1.0, beta=2.0).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'HybridPlusModelV2 | stages=7  filters=64ch  params={total_params:,}')

    # Train
    checkpoint_path = checkpoints_dir / 'hybrid_plus_v2.pt'
    print('\n--- Training Hybrid++ V2 ---')
    result = train_hybrid_plus_v2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        looks_options=list(LOOK_LEVELS),
        checkpoint_path=checkpoint_path,
        config=train_config,
    )

    # Evaluate
    print('\n--- Evaluating on test set ---')
    test_summary = evaluate_test_set(model, test_loader, device, results_root)

    # Save summary
    summary = {
        'device': str(device),
        'model': 'HybridPlus_V2',
        'num_stages': 7,
        'filters_per_bank': 32,
        'total_channels': 64,
        'seed': args.seed,
        'total_params': total_params,
        'best_val_loss': result.get('best_val_loss'),
        'test_evaluation': test_summary,
    }
    save_json(results_root / 'summary.json', summary)
    print(f'\nAll outputs saved to {results_root.resolve()}')


if __name__ == '__main__':
    main()
