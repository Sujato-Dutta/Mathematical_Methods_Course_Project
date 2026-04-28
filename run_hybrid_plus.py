from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import build_loader, discover_images, split_dataset
from src.hybrid_plus import HybridPlusModel
from src.losses import compute_loss
from src.noise import LOOK_LEVELS, add_gamma_noise
from src.train import TrainingConfig, evaluate_validation_loss, load_checkpoint, save_checkpoint
from src.utils import ensure_dir, get_device, save_json, set_seed


def train_hybrid_plus(
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

    # Train stage by stage
    for stage_index in range(len(model.stages)):
        # Freeze previous stages
        for prior_stage in model.stages[:stage_index]:
            for parameter in prior_stage.parameters():
                parameter.requires_grad = False

        # Unfreeze current stage
        for parameter in model.stages[stage_index].parameters():
            parameter.requires_grad = True

        # Ensure PDE parameters are always learnable across stages
        for parameter in model.pde.parameters():
            parameter.requires_grad = True

        # Ensure LevelEmbedder is learnable
        for parameter in model.level_embedder.parameters():
            parameter.requires_grad = True

        optimizer = torch.optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=config.learning_rate,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=config.scheduler_patience, factor=config.scheduler_factor,
        )

        best_val = float('inf')
        best_state = None
        patience = 0
        stage_history = {'train_loss': [], 'val_loss': []}

        epoch_bar = tqdm(range(config.epochs), desc=f'Training Hybrid++ Stage={stage_index + 1}', leave=False)
        for _ in epoch_bar:
            model.train()
            total_train = 0.0
            total_count = 0

            for batch in train_loader:
                clean = batch['image'].to(device)

                # Single model handles both L=1 and L=10
                for looks in looks_options:
                    noisy = add_gamma_noise(clean, looks)
                    estimate = model(noisy, looks, upto_stage=stage_index + 1)

                    loss = compute_loss(estimate, clean)

                    optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping max_norm=1.0
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Hybrid++ Model')
    parser.add_argument('--foe_path', default='./data/FoETrainingSets180', help='Path to FoETrainingSets180 grayscale images.')
    parser.add_argument('--results_dir', default='./results_hybrid_plus', help='Directory where Hybrid++ outputs are saved.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = ensure_dir(Path(args.results_dir))
    checkpoints_dir = ensure_dir(results_root / 'checkpoints')

    set_seed(args.seed)
    device = get_device()
    print(f'torch.cuda.is_available() = {device.type == "cuda"}')
    print(f'Using device: {device}')

    # Dataset — same split logic as run_pipeline.py
    foe_paths = discover_images(args.foe_path)
    foe_split = split_dataset(foe_paths, seed=args.seed)
    print(
        f'FoE split | total={len(foe_paths)} train={len(foe_split.train)} '
        f'val={len(foe_split.val)} test={len(foe_split.test)}'
    )

    train_config = TrainingConfig()
    train_loader = build_loader(foe_split.train, batch_size=train_config.batch_size, shuffle=True, augment=True)
    val_loader = build_loader(foe_split.val, batch_size=1, shuffle=False, augment=False)

    # Initialize Hybrid++ Model (PDE alpha/beta at default tuned values)
    model = HybridPlusModel(num_stages=10, alpha=1.0, beta=2.0).to(device)

    checkpoint_path = checkpoints_dir / 'hybrid_plus_model.pt'
    print('Training Hybrid++...')
    result = train_hybrid_plus(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        looks_options=list(LOOK_LEVELS),
        checkpoint_path=checkpoint_path,
        config=train_config,
    )

    summary = {
        'device': str(device),
        'model': 'HybridPlus',
        'num_stages': 10,
        'seed': args.seed,
        'best_val_loss': result.get('best_val_loss'),
    }
    save_json(results_root / 'summary.json', summary)
    print(f'Hybrid++ training complete. All artifacts saved to {results_root.resolve()}')


if __name__ == '__main__':
    main()
