# Hybrid++: Bridging Physics-Informed PDE Models and Deep Learning for Multiplicative Gamma Noise Removal

> A novel trainable nonlinear reaction-diffusion architecture that achieves near-DnCNN performance with **12.5× fewer parameters** while maintaining full mathematical interpretability.

## Key Results

| Method | Params | L=1 PSNR (dB) | L=10 PSNR (dB) |
|--------|--------|---------------|----------------|
| BM3D | 0 | 12.67 | 24.64 |
| PDE Baseline | 0 | 17.66 | 23.07 |
| TNRD | ~21k | 18.55 | 23.17 |
| **Hybrid++ (Ours)** | **~44k** | **20.86** | **25.02** |
| DnCNN | ~556k | 21.09 | 25.94 |

Hybrid++ closes the gap to DnCNN to within **0.23 dB** at L=1 using less than 8% of DnCNN's parameters.

## Architecture

Hybrid++ combines:
1. **Differentiable PDE Initialization** — Learnable α, β parameters optimized end-to-end
2. **64-Channel Multiscale Filter Banks** — 32 filters (3×3) + 32 filters (5×5) per stage
3. **4-Layer SE-Attention Influence Function** — Deep nonlinear diffusion shaping with channel attention (64→16→64)
4. **64-Dim Noise Level Embedding** — 3-layer MLP enabling blind denoising across noise levels
5. **Two-Phase Training** — Stage-wise initialization + joint end-to-end refinement

## Project Structure

```
├── src/
│   ├── hybrid_plus_v2.py        # Hybrid++ model architecture
│   ├── benchmark_models.py      # DnCNN baseline
│   ├── tnrd_model.py            # Original TNRD
│   ├── pde_baseline.py          # Classical PDE denoiser
│   ├── noise.py                 # Gamma noise generation
│   ├── losses.py                # MSE loss
│   ├── metrics.py               # PSNR / SSIM computation
│   ├── dataset.py               # FoE dataset loader
│   ├── train.py                 # Training utilities
│   └── utils.py                 # General utilities
├── paper/
│   └── main.tex                 # LaTeX paper source
├── figures/                     # Generated paper figures
├── data/
│   ├── FoETrainingSets180/      # FoE training images
│   └── SAR/                     # Real SAR test images
├── results/                     # Baseline model results & checkpoints
│   ├── results/checkpoints/     # DnCNN, TNRD, Hybrid checkpoints
│   └── BM3D/                    # BM3D evaluation results
├── results_hybrid_plus_v2_finetuned/  # Final Hybrid++ results
│   ├── checkpoints/             # stagewise.pt & finetuned.pt
│   ├── metrics_L1.csv           # Per-image metrics (L=1)
│   ├── metrics_L10.csv          # Per-image metrics (L=10)
│   └── summary.json             # Aggregate PSNR/SSIM
├── results_SAR/                 # SAR denoising results (qualitative)
├── run_finetuned_v2.py          # Train + finetune Hybrid++ (main script)
├── run_pipeline.py              # Train all baseline models
├── evaluate_bm3d.py             # BM3D evaluation on FoE test set
├── evaluate_sar_all.py          # SAR denoising with all models
├── generate_paper_figures.py    # Generate paper figures
└── requirements.txt
```

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
pip install bm3d  # For BM3D evaluation only
```

### Train Hybrid++
```bash
python run_finetuned_v2.py
```
This runs the complete two-phase pipeline:
- **Phase 1:** Stage-wise training (3 stages, lr=1e-3, early stopping)
- **Phase 2:** Joint fine-tuning (50 epochs, lr=1e-4, CosineAnnealingLR)

Results are saved to `./results_hybrid_plus_v2_finetuned/`.

### Train Baselines
```bash
python run_pipeline.py  # Trains PDE, TNRD, DnCNN
```

### Evaluate BM3D
```bash
python evaluate_bm3d.py
```

### Evaluate on Real SAR Images
```bash
python evaluate_sar_all.py
```

### Generate Paper Figures
```bash
python generate_paper_figures.py
```

## Citation

If you use this code, please cite:
```bibtex
@article{jetta2025hybridpp,
  title={Hybrid++: Bridging Physics-Informed PDE Models and Deep Learning for Multiplicative Gamma Noise Removal},
  author={Jetta, Mahipal and Dutta, Sujato},
  year={2025}
}
```

## License

This project is for academic research purposes.
