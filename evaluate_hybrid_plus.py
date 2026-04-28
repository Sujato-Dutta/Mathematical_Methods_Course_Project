import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.dataset import build_loader, discover_images, split_dataset
from src.hybrid_plus import HybridPlusModel
from src.metrics import compute_psnr, compute_ssim
from src.noise import LOOK_LEVELS, add_gamma_noise
from src.train import load_checkpoint
from src.utils import ensure_dir, get_device, set_seed


@torch.no_grad()
def evaluate_hybrid_plus_model():
    parser = argparse.ArgumentParser(description='Evaluate trained Hybrid++ Model')
    parser.add_argument('--foe_path', default='./data/FoETrainingSets180', help='Path to dataset')
    parser.add_argument('--results_dir', default='./results_hybrid_plus', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f'Using device: {device}')

    results_root = Path(args.results_dir)
    checkpoint_path = results_root / 'checkpoints' / 'hybrid_plus_model.pt'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Ensure the model is trained.")

    # Load dataset
    foe_paths = discover_images(args.foe_path)
    foe_split = split_dataset(foe_paths, seed=args.seed)
    test_loader = build_loader(foe_split.test, batch_size=1, shuffle=False, augment=False)
    
    # Initialize and load model
    model = HybridPlusModel(num_stages=10, alpha=1.0, beta=2.0).to(device)
    print(f"Loading checkpoint from {checkpoint_path}...")
    load_checkpoint(checkpoint_path, model, device)
    model.eval()

    metric_rows = []

    for looks in LOOK_LEVELS:
        for batch in tqdm(test_loader, desc=f'Evaluating Hybrid++ L={looks}'):
            clean = batch['image'].to(device)
            name = str(batch['name'][0])
            
            noisy = add_gamma_noise(clean, looks)
            estimate = model(noisy, looks)
            
            psnr_val = compute_psnr(clean, estimate)
            ssim_val = compute_ssim(clean, estimate)
            
            metric_rows.append({
                'dataset': 'foe',
                'image': name,
                'looks': looks,
                'model': 'Hybrid++',
                'psnr': psnr_val,
                'ssim': ssim_val
            })

    # Save raw metric rows
    csv_path = results_root / 'hybrid_plus_test_metrics.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset', 'image', 'looks', 'model', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(metric_rows)

    # Compute aggregations
    grouped = defaultdict(list)
    for row in metric_rows:
        grouped[row['looks']].append(row)

    print("\n--- Final Hybrid++ Test Set Evaluation ---")
    summary_results = []
    
    for looks in LOOK_LEVELS:
        subset = grouped.get(looks, [])
        if not subset:
            continue
            
        psnr_values = np.array([float(r['psnr']) for r in subset])
        ssim_values = np.array([float(r['ssim']) for r in subset])
        
        psnr_mean, psnr_std = psnr_values.mean(), psnr_values.std()
        ssim_mean, ssim_std = ssim_values.mean(), ssim_values.std()
        
        print(f"L={looks}: PSNR = {psnr_mean:.4f} ± {psnr_std:.4f} | SSIM = {ssim_mean:.4f} ± {ssim_std:.4f}")
        
        summary_results.append({
            'looks': looks,
            'psnr_mean': float(psnr_mean),
            'psnr_std': float(psnr_std),
            'ssim_mean': float(ssim_mean),
            'ssim_std': float(ssim_std),
        })

    # Update summary.json if it exists
    summary_path = results_root / 'summary.json'
    if summary_path.exists():
        with summary_path.open('r', encoding='utf-8') as f:
            summary = json.load(f)
            
        summary['test_evaluation'] = summary_results
        
        with summary_path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    print(f"\nEvaluation saved to: {csv_path} and {summary_path}")


if __name__ == '__main__':
    evaluate_hybrid_plus_model()
