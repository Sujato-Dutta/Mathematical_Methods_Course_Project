from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import compute_psnr, compute_ssim, summarize_metric_rows
from src.noise import add_gamma_noise
from src.pde_baseline import nonlinear_smooth_diffusion_denoise
from src.utils import ensure_dir, save_image, tensor_to_numpy


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    looks: int,
    output_root: str | Path,
) -> dict[str, object]:
    output_root = Path(output_root)
    image_dir = ensure_dir(output_root / "denoised_images" / f"L{looks}")
    figure_dir = ensure_dir(output_root / "figures")

    metric_rows: list[dict[str, float | str | int]] = []
    collected_examples: list[dict[str, torch.Tensor | str]] = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating L={looks}", leave=False):
            clean = batch["image"].to(device)
            name = str(batch["name"][0])
            noisy = add_gamma_noise(clean, looks=looks)
            pde = nonlinear_smooth_diffusion_denoise(noisy, looks=looks)
            tnrd = model(noisy, looks=looks)

            save_image(image_dir / f"{name}_noisy.png", noisy)
            save_image(image_dir / f"{name}_pde.png", pde)
            save_image(image_dir / f"{name}_tnrd.png", tnrd)

            metric_rows.extend(
                [
                    {
                        "image": name,
                        "looks": looks,
                        "method": "Noisy",
                        "psnr": compute_psnr(clean, noisy),
                        "ssim": compute_ssim(clean, noisy),
                    },
                    {
                        "image": name,
                        "looks": looks,
                        "method": "PDE",
                        "psnr": compute_psnr(clean, pde),
                        "ssim": compute_ssim(clean, pde),
                    },
                    {
                        "image": name,
                        "looks": looks,
                        "method": "TNRD",
                        "psnr": compute_psnr(clean, tnrd),
                        "ssim": compute_ssim(clean, tnrd),
                    },
                ]
            )

            if not collected_examples:
                collected_examples.append(
                    {"name": name, "clean": clean.cpu(), "noisy": noisy.cpu(), "pde": pde.cpu(), "tnrd": tnrd.cpu()}
                )

    metrics_path = output_root / "tables" / f"metrics_L{looks}.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "looks", "method", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(metric_rows)

    if collected_examples:
        example = collected_examples[0]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for axis, key, title in zip(
            axes,
            ["clean", "noisy", "pde", "tnrd"],
            ["Clean", "Noisy", "PDE", "TNRD"],
        ):
            axis.imshow(tensor_to_numpy(example[key]), cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(title)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(figure_dir / f"comparison_L{looks}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    summary_rows = []
    for method in ("Noisy", "PDE", "TNRD"):
        subset = [row for row in metric_rows if row["method"] == method]
        summary_rows.append(
            {
                "method": method,
                "psnr": float(sum(float(row["psnr"]) for row in subset) / len(subset)),
                "ssim": float(sum(float(row["ssim"]) for row in subset) / len(subset)),
            }
        )
    return {
        "rows": metric_rows,
        "summary": summarize_metric_rows(metric_rows),
        "table": summary_rows,
    }


def plot_psnr_comparison(metrics_csv_paths: list[str | Path], output_path: str | Path) -> None:
    grouped: dict[tuple[int, str], list[float]] = {}
    for path in metrics_csv_paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (int(row["looks"]), row["method"])
                grouped.setdefault(key, []).append(float(row["psnr"]))

    fig, ax = plt.subplots(figsize=(8, 4))
    for looks in sorted({key[0] for key in grouped}):
        methods = [method for group_looks, method in grouped if group_looks == looks]
        values = [sum(grouped[(looks, method)]) / len(grouped[(looks, method)]) for method in methods]
        ax.bar(
            [f"{method}\nL={looks}" for method in methods],
            values,
            label=f"L={looks}",
            alpha=0.8,
        )
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Average PSNR Comparison")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
