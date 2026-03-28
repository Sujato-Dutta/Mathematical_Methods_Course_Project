from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.ablation import run_ablation_study
from src.dataset import GrayscaleImageDataset, discover_images, split_dataset
from src.evaluate import evaluate_model, plot_psnr_comparison
from src.tnrd_model import TNRDModel
from src.train import TrainingConfig, train_model
from src.utils import ensure_dir, get_device, save_json, set_seed


def build_loaders(data_root: Path, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    split = split_dataset(discover_images(data_root))
    train_loader = DataLoader(GrayscaleImageDataset(split.train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GrayscaleImageDataset(split.val), batch_size=1, shuffle=False)
    test_loader = DataLoader(GrayscaleImageDataset(split.test), batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader


def write_paper() -> None:
    paper_dir = ensure_dir("paper")
    tex = rf"""\documentclass[conference]{{IEEEtran}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{url}}
\title{{Trainable Nonlinear Reaction Diffusion for Multiplicative Gamma Noise Removal}}
\author{{Automated Research Pipeline}}
\begin{{document}}
\maketitle
\begin{{abstract}}
This paper presents an end-to-end implementation of trainable nonlinear reaction diffusion for grayscale image restoration under multiplicative gamma noise. The pipeline trains stage-wise diffusion models with frozen linear filters and learned influence functions, compares against a nonlinear smooth diffusion PDE baseline, performs ablations, and exports paper-ready visualizations.
\end{{abstract}}
\section{{Introduction}}
Multiplicative gamma noise appears in coherent imaging systems and is harder to remove than additive Gaussian corruption because the observation model depends on the signal magnitude. Trainable nonlinear reaction diffusion (TNRD) provides a compact unrolled optimization view that is fast at inference time and interpretable through stage-wise diffusion updates.
\section{{Related Work}}
Reaction diffusion methods bridge PDE-based restoration and discriminative learning. The comparison baseline in this project is a nonlinear smooth diffusion model guided by gray-level and gradient information, while TNRD learns influence functions and reaction weights directly from data for improved restoration quality.
\section{{Mathematical Formulation}}
We implement the update rule
\begin{{equation}}
u_t = u_{{t-1}} - \left(\sum_{{i=1}}^{{N_k}} \bar{{k}}_i^t \left( \frac{{u_\sigma}}{{M}} \phi_i^t(k_i^t * u_{{(t-1)p}}) \right) + \lambda \left( \frac{{u-f}}{{u^2 + \varepsilon}} \right)\right),
\end{{equation}}
where $u_t$ is the restored image at stage $t$, $f$ is the noisy image, $M$ is the gamma looks parameter, and $u_\sigma$ is obtained by Gaussian smoothing. The filters $k_i$ are randomly initialized and frozen, while influence functions $\phi_i$ and scalar $\lambda$ are learned.
\section{{Proposed Method}}
The model uses $N_k=8$ filters of size $3\times3$ and five stages. Each stage applies fixed convolutions, a learned influence function implemented as $1\times1$ convolution followed by $\tanh$, and a learned reaction weight. Training minimizes mean squared error on grayscale images discovered recursively under the data directory and corrupted with multiplicative gamma noise at $L \in \{{1,10\}}$.
\section{{Experimental Setup}}
The dataset is split into train, validation, and test subsets from the grayscale images found recursively under the data directory. Adam is used with learning rate $10^{{-3}}$, batch size 1, and 30 epochs. We evaluate PSNR and SSIM for noisy inputs, nonlinear smooth diffusion PDE outputs, and TNRD outputs.
\section{{Results}}
Figure~\ref{{fig:comparison}} shows qualitative restoration. Figure~\ref{{fig:psnr}} summarizes PSNR across methods and noise levels.
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../results/figures/comparison_L1.png}}
\caption{{Qualitative comparison for $L=1$.}}
\label{{fig:comparison}}
\end{{figure}}
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../results/figures/psnr_comparison.png}}
\caption{{Average PSNR comparison across methods and looks.}}
\label{{fig:psnr}}
\end{{figure}}
\section{{Ablation Study}}
Table~\ref{{tab:ablation}} reports the effect of removing nonlinear influence functions, reducing filters, and reducing stages.
\begin{{table}}[t]
\centering
\caption{{Ablation study for $L=1$.}}
\label{{tab:ablation}}
\input{{../results/tables/ablation_L1.tex}}
\end{{table}}
\section{{Conclusion}}
The automated pipeline reproduces a practical TNRD variant for multiplicative gamma noise removal, integrates a nonlinear smooth diffusion PDE baseline, and generates executable experimental artifacts and a paper-ready report.
\bibliographystyle{{IEEEtran}}
\begin{{thebibliography}}{{1}}
\bibitem{{chen2017tnrd}}
Y. Chen and T. Pock, ``Trainable nonlinear reaction diffusion: A flexible framework for fast and effective image restoration,'' \emph{{IEEE Transactions on Pattern Analysis and Machine Intelligence}}, vol. 39, no. 6, pp. 1256--1272, 2017.
\end{{thebibliography}}
\end{{document}}
"""
    (paper_dir / "main.tex").write_text(tex, encoding="utf-8")


def export_ablation_table(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        '\\begin{tabular}{lcccccc}',
        '\\toprule',
        'Variant & L & Filters & Stages & Nonlinear & PSNR & SSIM \\\\',
        '\\midrule',
    ]
    for row in rows:
        lines.append(
            f"{row['variant'].replace('_', ' ')} & {row['looks']} & {row['num_filters']} & {row['num_stages']} & {row['use_nonlinearity']} & {float(row['psnr']):.4f} & {float(row['ssim']):.4f} \\\\"
        )
    lines.extend(['\\bottomrule', '\\end{tabular}'])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    results_root = ensure_dir(project_root / "results")
    ensure_dir(results_root / "figures")
    ensure_dir(results_root / "tables")
    ensure_dir(results_root / "denoised_images")
    ensure_dir(results_root / "models")
    set_seed(123)
    device = get_device()

    train_loader, val_loader, test_loader = build_loaders(project_root / "data", batch_size=1)

    metrics_csv_paths: list[Path] = []
    summary: dict[str, object] = {"device": str(device), "looks": {}}
    for looks in (1, 10):
        model = TNRDModel(num_filters=8, kernel_size=3, num_stages=5, use_nonlinearity=True).to(device)
        checkpoint_path = results_root / "models" / f"tnrd_L{looks}.pt"
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=TrainingConfig(epochs=30, learning_rate=1e-3, batch_size=1, looks=looks),
            model_path=checkpoint_path,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        evaluation = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            looks=looks,
            output_root=results_root,
        )
        metrics_csv_paths.append(results_root / "tables" / f"metrics_L{looks}.csv")
        summary["looks"][f"L{looks}"] = {"training_history": history, "evaluation": evaluation["table"]}

        ablation_rows = run_ablation_study(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            looks=looks,
            output_root=results_root,
        )
        export_ablation_table(ablation_rows, results_root / "tables" / f"ablation_L{looks}.tex")

    plot_psnr_comparison(metrics_csv_paths, results_root / "figures" / "psnr_comparison.png")
    save_json(results_root / "summary.json", summary)
    write_paper()


if __name__ == "__main__":
    main()
