from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import compute_loss
from src.noise import LOOK_LEVELS, add_gamma_noise
from src.pde_baseline import PDEConfig, nonlinear_smooth_diffusion_denoise


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 4
    early_stopping_patience: int = 10
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5


def save_checkpoint(path: Path, model: nn.Module, metadata: dict[str, object]) -> None:
    torch.save({'model_state_dict': model.state_dict(), **metadata}, path)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def evaluate_validation_loss(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    looks_options: Sequence[int],
    upto_stage: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in val_loader:
            clean = batch['image'].to(device)
            for looks in looks_options:
                noisy = add_gamma_noise(clean, looks)
                estimate = model(noisy, looks, upto_stage=upto_stage) if upto_stage is not None else model(noisy, looks)
                total_loss += float(compute_loss(estimate, clean).item())
                total_count += 1
    return total_loss / max(1, total_count)


def train_standard_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    looks_options: Sequence[int],
    model_name: str,
    checkpoint_path: Path,
    config: TrainingConfig,
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, model, device)
        return {'best_val_loss': checkpoint.get('best_val_loss'), 'resumed_from_checkpoint': True}

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=config.scheduler_patience, factor=config.scheduler_factor)
    best_val = float('inf')
    best_state = None
    patience = 0
    history = {'train_loss': [], 'val_loss': []}

    epoch_bar = tqdm(range(config.epochs), desc=f'Training {model_name}', leave=False)
    for _ in epoch_bar:
        model.train()
        total_train = 0.0
        total_count = 0
        for batch in train_loader:
            clean = batch['image'].to(device)
            for looks in looks_options:
                noisy = add_gamma_noise(clean, looks)
                estimate = model(noisy, looks)
                loss = compute_loss(estimate, clean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train += float(loss.item())
                total_count += 1

        mean_train = total_train / max(1, total_count)
        mean_val = evaluate_validation_loss(model, val_loader, device, looks_options)
        scheduler.step(mean_val)
        history['train_loss'].append(mean_train)
        history['val_loss'].append(mean_val)
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
    save_checkpoint(checkpoint_path, model, {'best_val_loss': best_val, 'looks_options': list(looks_options), 'history': history})
    return {'best_val_loss': best_val, 'history': history}


def train_stagewise_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    looks: int,
    model_name: str,
    checkpoint_path: Path,
    config: TrainingConfig,
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, model, device)
        return {'best_val_loss': checkpoint.get('best_val_loss'), 'resumed_from_checkpoint': True}

    final_best_val = float('inf')
    stage_histories: list[dict[str, object]] = []
    for stage_index in range(len(model.stages)):
        for prior_stage in model.stages[:stage_index]:
            for parameter in prior_stage.parameters():
                parameter.requires_grad = False
        for parameter in model.stages[stage_index].parameters():
            parameter.requires_grad = True

        optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=config.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=config.scheduler_patience, factor=config.scheduler_factor)
        best_val = float('inf')
        best_state = None
        patience = 0
        stage_history = {'train_loss': [], 'val_loss': []}

        epoch_bar = tqdm(range(config.epochs), desc=f'Training {model_name} Stage={stage_index + 1}', leave=False)
        for _ in epoch_bar:
            model.train()
            total_train = 0.0
            total_count = 0
            for batch in train_loader:
                clean = batch['image'].to(device)
                noisy = add_gamma_noise(clean, looks)
                estimate = model(noisy, looks, upto_stage=stage_index + 1)
                loss = compute_loss(estimate, clean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train += float(loss.item())
                total_count += 1

            mean_train = total_train / max(1, total_count)
            mean_val = evaluate_validation_loss(model, val_loader, device, [looks], upto_stage=stage_index + 1)
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

    save_checkpoint(checkpoint_path, model, {'best_val_loss': final_best_val, 'looks_options': [looks], 'stage_histories': stage_histories})
    return {'best_val_loss': final_best_val, 'stage_histories': stage_histories}


def tune_pde(val_loader: DataLoader, device: torch.device) -> PDEConfig:
    alphas = (0.5, 1.0, 1.5, 2.0)
    betas = (1.0, 1.5, 2.0, 2.5, 3.0)
    best_config = PDEConfig()
    best_loss = float('inf')
    print('Tuning PDE baseline on FoE validation split with MSE.')
    for alpha in alphas:
        for beta in betas:
            config = PDEConfig(alpha=alpha, beta=beta)
            total_loss = 0.0
            total_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    clean = batch['image'].to(device)
                    for looks in LOOK_LEVELS:
                        noisy = add_gamma_noise(clean, looks)
                        estimate = nonlinear_smooth_diffusion_denoise(noisy, looks=looks, config=config)
                        total_loss += float(compute_loss(estimate, clean).item())
                        total_count += 1
            mean_loss = total_loss / max(1, total_count)
            print(f'PDE tune | alpha={alpha:.2f} beta={beta:.2f} val_mse={mean_loss:.6f}')
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_config = config
    print(
        f'Selected PDE config | alpha={best_config.alpha:.2f} beta={best_config.beta:.2f} '
        f'delta_t={best_config.delta_t:.2f} iterations={best_config.num_iterations}'
    )
    return best_config
