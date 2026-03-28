# tnrd_gamma_denoising

End-to-end research pipeline for multiplicative gamma noise removal in grayscale images. The central research contribution in this project is a Trainable Nonlinear Reaction Diffusion (TNRD) model, and its performance is compared against a nonlinear smooth diffusion PDE baseline to study the improvements achieved by the learned approach.

## Project Overview

This project is built around two models:

1. Proposed research model: Trainable Nonlinear Reaction Diffusion (TNRD)
2. Comparison model: Nonlinear Smooth Diffusion PDE baseline

The objective is to evaluate whether the proposed TNRD formulation improves restoration quality over the PDE baseline for multiplicative gamma noise at `L=1` and `L=10`.

The full pipeline covers:

1. loading grayscale images from the dataset
2. synthesizing multiplicative gamma noise
3. training the proposed TNRD model stage-wise
4. evaluating the PDE baseline and the trained TNRD model
5. computing PSNR and SSIM metrics
6. running ablation studies on the TNRD model
7. generating figures, tables, denoised images, and a LaTeX report

Everything runs from a single command:

```bash
python run_pipeline.py
```

## Implemented Models

### 1. Proposed Model: Trainable Nonlinear Reaction Diffusion (TNRD)

The TNRD model is implemented in [src/tnrd_model.py](C:\Users\Sujato\Downloads\SEM VI\Mathematical Methods\Project\tnrd_gamma_denoising\src\tnrd_model.py).

This is the main research model in the project. It follows the update rule:

```text
u_t = u_{t-1} - (
    sum_i kbar_i^t ( (u_sigma / M) * phi_i^t(k_i^t * u_(t-1)p) )
    + lambda * ((u - f) / (u^2 + epsilon))
)
```

Key properties:

1. `Nk = 8` filters
2. filter size `3x3`
3. `5` stages
4. filters are randomly initialized and frozen
5. only stage-wise influence functions and reaction parameters are learned
6. Gaussian smoothing is used to compute `u_sigma`

The ablation study is centered on this model to analyze which TNRD components contribute most to performance.

### 2. Comparison Baseline: Nonlinear Smooth Diffusion PDE

The PDE baseline is implemented in [src/pde_baseline.py](C:\Users\Sujato\Downloads\SEM VI\Mathematical Methods\Project\tnrd_gamma_denoising\src\pde_baseline.py).

This model is used as the comparison baseline for the proposed TNRD approach. It follows the nonlinear diffusion model:

```text
du/dt = div(g(u_sigma, |grad u_sigma|) grad u)
```

with diffusion coefficient:

```text
g(u_sigma, |grad u_sigma|) = (u_sigma / M)^alpha / (1 + |grad u_sigma|^beta)
```

This baseline uses:

1. gray-level guidance through `u_sigma`
2. edge guidance through `|grad u_sigma|`
3. Gaussian smoothing for stable diffusion
4. replicated boundary handling to approximate zero normal flux

## Dataset

The dataset loader is implemented in [src/dataset.py](C:\Users\Sujato\Downloads\SEM VI\Mathematical Methods\Project\tnrd_gamma_denoising\src\dataset.py).

The pipeline recursively discovers grayscale image files under `data/`.

Current dataset layout:

```text
data/
  FoETrainingSets180/
    test_001.png
    ...
    test_400.png
```

All supported image files are split automatically into:

1. 80% training
2. 10% validation
3. 10% testing

## Noise Model

Multiplicative gamma noise is generated as:

```text
f = u * n
n ~ Gamma(L, L)
```

The project evaluates two noise levels:

1. `L = 1`
2. `L = 10`

## Training and Evaluation

Training is implemented in [src/train.py](C:\Users\Sujato\Downloads\SEM VI\Mathematical Methods\Project\tnrd_gamma_denoising\src\train.py).
Evaluation is implemented in [src/evaluate.py](C:\Users\Sujato\Downloads\SEM VI\Mathematical Methods\Project\tnrd_gamma_denoising\src\evaluate.py).
Ablation experiments are implemented in [src/ablation.py](C:\Users\Sujato\Downloads\SEM VI\Mathematical Methods\Project\tnrd_gamma_denoising\src\ablation.py).

Training settings for the proposed TNRD model:

1. loss: MSE
2. optimizer: Adam
3. learning rate: `1e-3`
4. batch size: `1`
5. epochs: `30`
6. stage-wise optimization

Evaluation compares:

1. noisy image
2. nonlinear smooth diffusion PDE baseline output
3. proposed TNRD output

Metrics:

1. PSNR
2. SSIM

Progress bars are shown during:

1. training
2. evaluation
3. ablation runs

## Project Structure

```text
tnrd_gamma_denoising/
│
├── data/
├── src/
│   ├── dataset.py
│   ├── noise.py
│   ├── tnrd_model.py
│   ├── pde_baseline.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   ├── ablation.py
│   └── utils.py
│
├── results/
│   ├── figures/
│   ├── tables/
│   ├── denoised_images/
│   └── models/
│
├── paper/
├── requirements.txt
├── run_pipeline.py
└── README.md
```

## Installation

Using the local virtual environment:

```bash
venv\Scripts\python.exe -m pip install -r requirements.txt
```

Or with the active Python environment:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Run the complete project with:

```bash
venv\Scripts\python.exe run_pipeline.py
```

or:

```bash
python run_pipeline.py
```

The pipeline will:

1. load the dataset from `data/`
2. create noisy observations for `L=1` and `L=10`
3. train the proposed TNRD model
4. evaluate the nonlinear smooth diffusion PDE baseline
5. compute metrics
6. run TNRD ablation experiments
7. save figures, tables, denoised outputs, and model checkpoints
8. generate `paper/main.tex`

## Outputs

Generated results are stored in `results/`:

1. `results/models/` for saved checkpoints
2. `results/denoised_images/` for noisy, PDE, and TNRD outputs
3. `results/figures/` for comparison images and PSNR charts
4. `results/tables/` for metrics and ablation tables
5. `results/summary.json` for experiment summaries

## Research Report

The LaTeX project report is stored in `paper/main.tex`.

To compile in Overleaf:

1. upload the `paper/` folder contents
2. upload the `results/` folder
3. set `main.tex` as the main file
4. compile with `pdfLaTeX`

Figure and table paths are already written relative to `paper/`, for example:

```latex
\includegraphics[width=\linewidth]{../results/figures/comparison_L1.png}
```
