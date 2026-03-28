from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.noise import add_gamma_noise


@dataclass
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    batch_size: int = 1
    looks: int = 1


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    model_path: str | Path,
) -> dict[str, list[float]]:
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    model_path = Path(model_path)

    for stage_index in range(1, len(model.stages) + 1):
        for prior_stage in model.stages[: stage_index - 1]:
            for parameter in prior_stage.parameters():
                parameter.requires_grad = False
        for parameter in model.stages[stage_index - 1].parameters():
            parameter.requires_grad = True

        optimizer = torch.optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=config.learning_rate,
        )

        epoch_bar = tqdm(
            range(config.epochs),
            desc=f"Training L={config.looks} Stage={stage_index}",
            leave=False,
        )
        for _ in epoch_bar:
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                clean = batch["image"].to(device)
                noisy = add_gamma_noise(clean, looks=config.looks)
                denoised = model(noisy, looks=config.looks, upto_stage=stage_index)
                loss = criterion(denoised, clean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    clean = batch["image"].to(device)
                    noisy = add_gamma_noise(clean, looks=config.looks)
                    denoised = model(noisy, looks=config.looks, upto_stage=stage_index)
                    val_loss += criterion(denoised, clean).item()

            mean_train = train_loss / max(1, len(train_loader))
            mean_val = val_loss / max(1, len(val_loader))
            history["train_loss"].append(mean_train)
            history["val_loss"].append(mean_val)
            epoch_bar.set_postfix(train=f"{mean_train:.4f}", val=f"{mean_val:.4f}")

            if stage_index == len(model.stages) and mean_val < best_val:
                best_val = mean_val
                torch.save({"model_state_dict": model.state_dict(), "looks": config.looks}, model_path)

    if not model_path.exists():
        torch.save({"model_state_dict": model.state_dict(), "looks": config.looks}, model_path)

    return history
