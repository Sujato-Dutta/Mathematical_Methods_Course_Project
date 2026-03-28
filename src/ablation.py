from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import compute_psnr, compute_ssim
from src.noise import add_gamma_noise
from src.tnrd_model import TNRDModel
from src.train import TrainingConfig, train_model
from src.utils import ensure_dir


ABLATIONS = [
    {"name": "full_model", "num_filters": 8, "num_stages": 5, "use_nonlinearity": True},
    {"name": "no_nonlinear_influence", "num_filters": 8, "num_stages": 5, "use_nonlinearity": False},
    {"name": "nk_4_filters", "num_filters": 4, "num_stages": 5, "use_nonlinearity": True},
    {"name": "only_3_stages", "num_filters": 8, "num_stages": 3, "use_nonlinearity": True},
]


def _load_existing_rows(csv_path: Path) -> dict[str, dict[str, float | str | int | bool]]:
    if not csv_path.exists():
        return {}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            rows[str(row["variant"])] = {
                "variant": row["variant"],
                "looks": int(row["looks"]),
                "num_filters": int(row["num_filters"]),
                "num_stages": int(row["num_stages"]),
                "use_nonlinearity": row["use_nonlinearity"].lower() == "true",
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
            }
        return rows


def run_ablation_study(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    looks: int,
    output_root: str | Path,
) -> list[dict[str, float | str | int | bool]]:
    output_root = Path(output_root)
    table_dir = ensure_dir(output_root / "tables")
    model_dir = ensure_dir(output_root / "models")
    csv_path = table_dir / f"ablation_L{looks}.csv"
    existing_rows = _load_existing_rows(csv_path)
    rows: list[dict[str, float | str | int | bool]] = []

    for spec in ABLATIONS:
        variant_name = spec["name"]
        if variant_name in existing_rows:
            print(f"Found existing ablation result for {variant_name} at L={looks}. Skipping.")
            rows.append(existing_rows[variant_name])
            continue

        model = TNRDModel(
            num_filters=spec["num_filters"],
            num_stages=spec["num_stages"],
            use_nonlinearity=spec["use_nonlinearity"],
        ).to(device)
        checkpoint_path = model_dir / f"ablation_{variant_name}_L{looks}.pt"

        if checkpoint_path.exists():
            print(f"Found existing ablation checkpoint for {variant_name} at L={looks}. Skipping retraining.")
        else:
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                config=TrainingConfig(looks=looks, epochs=10),
                model_path=checkpoint_path,
            )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        psnr_scores: list[float] = []
        ssim_scores: list[float] = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Ablation {variant_name} L={looks}", leave=False):
                clean = batch["image"].to(device)
                noisy = add_gamma_noise(clean, looks=looks)
                denoised = model(noisy, looks=looks)
                psnr_scores.append(compute_psnr(clean, denoised))
                ssim_scores.append(compute_ssim(clean, denoised))

        rows.append(
            {
                "variant": variant_name,
                "looks": looks,
                "num_filters": spec["num_filters"],
                "num_stages": spec["num_stages"],
                "use_nonlinearity": spec["use_nonlinearity"],
                "psnr": float(sum(psnr_scores) / len(psnr_scores)),
                "ssim": float(sum(ssim_scores) / len(ssim_scores)),
            }
        )

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["variant", "looks", "num_filters", "num_stages", "use_nonlinearity", "psnr", "ssim"],
            )
            writer.writeheader()
            writer.writerows(rows)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "looks", "num_filters", "num_stages", "use_nonlinearity", "psnr", "ssim"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows
