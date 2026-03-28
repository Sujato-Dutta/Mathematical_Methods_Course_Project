# TNRD Gamma Denoising

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

This is the main research model in the project. The current implementation follows:

$$ u_t = u_{t-1} - \left( \sum_{i=1}^{N_k} \bar{k}_i^t \left( \frac{u_\sigma}{M} \, \phi_i^t(k_i^t * u_{(t-1)p}) \right) + \lambda \left( \frac{u-f}{u^2 + \varepsilon} \right) \right) $$

Implementation mapping:

1. `u_{t-1}` is the current iterate stored as `u_prev`
2. `u_{(t-1)p}` is obtained by reflection padding before the convolution with `k_i^t`
3. `k_i^t * u_{(t-1)p}` is implemented with `F.conv2d`
4. `\bar{k}_i^t * (...)` is implemented with `F.conv_transpose2d` using flipped filters
5. `\phi_i^t` is implemented as a learnable grouped `1x1` convolution followed by `tanh`
6. `M` is implemented using the gamma parameter passed into the model, which is `L` in the experiments
7. the reaction term uses the current iterate `u = u_{t-1}` in `((u - f) / (u^2 + \varepsilon))`

Interpretation:

1. Start from the previous estimate `u_{t-1}`.
2. Apply the frozen filters `k_i^t` to the padded image.
3. Pass each filter response through the learnable influence function `\phi_i^t`.
4. Weight the diffusion term using the smoothed image `u_\sigma` and the gamma parameter `M`.
5. Add the reaction term that pulls the estimate toward the noisy observation `f`.
6. Subtract the full update to obtain the next stage output `u_t`.

Symbols used above:

1. `u_t` is the restored image at stage `t`
2. `f` is the noisy observation
3. `u_\sigma` is the Gaussian-smoothed image
4. `M` is the gamma noise parameter, taken as `L` in this project
5. `k_i^t` are the frozen convolution filters
6. `\bar{k}_i^t` are the flipped filters used in the reverse diffusion step
7. `\phi_i^t` are the learnable influence functions
8. `\lambda` is the learnable scalar regularization parameter that weights the reaction term

Key properties:

1. `Nk = 8` filters
2. filter size `3x3`
3. `5` stages
4. filters are randomly initialized and frozen
5. only stage-wise influence functions and scalar regularization parameters are learned
6. Gaussian smoothing is used to compute `u_\sigma`

The ablation study is centered on this model to analyze which TNRD components contribute most to performance.

### 2. Comparison Baseline: Nonlinear Smooth Diffusion PDE

This model is used as the comparison baseline for the proposed TNRD approach. It follows the nonlinear diffusion equation:

$$ \frac{\partial u}{\partial t} = \mathrm{div} \left( g(u_\sigma, |\nabla u_\sigma|) \, \nabla u \right) $$

with diffusion coefficient:

$$ g(u_\sigma, |\nabla u_\sigma|) = \left( \frac{u_\sigma}{M} \right)^{\alpha} \frac{1}{1 + |\nabla u_\sigma|^{\beta}} $$

Interpretation:

1. Smooth the current image using a Gaussian kernel to obtain `u_\sigma`.
2. Compute the gradient magnitude of the smoothed image.
3. Build a diffusion coefficient that depends on both gray-level information and edge information.
4. Diffuse strongly in smooth regions and more carefully near edges.
5. Repeat this process iteratively to denoise the image.

Symbols used above:

1. `u_\sigma = G_\sigma * u` is the Gaussian-smoothed image
2. `M = \max_x |u_\sigma(x,t)|` is the maximum gray level
3. `\alpha` controls gray-level influence
4. `\beta` controls edge sensitivity

Implementation mapping:

1. `u_\sigma = G_\sigma * u` is computed by Gaussian convolution on the current iterate
2. `|\nabla u_\sigma|` is approximated using four directional finite differences
3. `M` is computed as the maximum value of `u_\sigma` for each image
4. `\mathrm{div}(g\nabla u)` is approximated using directional fluxes and an explicit time update
5. replicated padding is used to approximate the zero-normal-flux boundary condition

## Dataset

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

$$ f = u \cdot n, \qquad n \sim \Gamma(L, L) $$

Interpretation:

1. `u` is the clean image
2. `n` is a random gamma-distributed noise variable
3. `f` is the observed noisy image obtained by multiplying `u` and `n`

The project evaluates two noise levels:

1. `L = 1`
2. `L = 10`

## Training and Evaluation

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
