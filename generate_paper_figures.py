import matplotlib.pyplot as plt
import torch
from pathlib import Path
import cv2
import numpy as np

def generate_performance_scatter():
    # Data for L=10
    models = ['PDE', 'TNRD', 'Hybrid++', 'DnCNN']
    params = [0, 21120, 44408, 556097]
    psnr = [23.07, 23.17, 25.02, 25.94]
    
    # We replace 0 with 1 for log scale plotting, or just don't plot PDE on log scale 
    # Let's use a linear scale but space it out, or symlog. Symlog is better.
    
    plt.figure(figsize=(10, 6))
    
    # Plot each point with a different color/marker
    colors = ['#ff9800', '#795548', '#d32f2f', '#2196f3']
    markers = ['o', 's', 'D', '^']
    
    for i in range(len(models)):
        plt.scatter(params[i], psnr[i], color=colors[i], marker=markers[i], s=200, label=models[i], zorder=5)
        # Add text labels slightly offset
        plt.text(params[i], psnr[i] - 0.15, models[i], fontsize=12, ha='center', va='top', fontweight='bold' if models[i] == 'Hybrid++' else 'normal')

    plt.xscale('symlog', linthresh=1000)
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.xlabel('Number of Learnable Parameters (Log Scale)', fontsize=14)
    plt.ylabel('PSNR (dB) at L=10', fontsize=14)
    plt.title('Performance vs. Model Capacity (FoE L=10)', fontsize=16, pad=15)
    
    plt.xlim(0, 1000000)
    plt.ylim(22.5, 26.5)
    
    plt.tight_layout()
    plt.savefig('fig_performance_vs_params.png', dpi=300)
    plt.close()

def generate_sar_comparison():
    sar_dir = Path('./results_SAR/denoised_L1')
    
    # Choose 3 images
    images = [
        'ROIs1868_summer_s1_59_p1000',
        'ROIs1868_summer_s1_59_p100',
        'ROIs1868_summer_s1_59_p10'
    ]
    
    models = ['original', 'PDE', 'TNRD', 'HybridPP', 'DnCNN']
    titles = ['Noisy SAR', 'PDE', 'TNRD', 'Hybrid++', 'DnCNN']
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    for row, img_name in enumerate(images):
        for col, model in enumerate(models):
            path = sar_dir / f"{img_name}_{model}.png"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                
                # Crop a 100x100 patch from the center to show details better
                h, w = img.shape
                cy, cx = h//2, w//2
                patch = img[cy-50:cy+50, cx-50:cx+50]
                
                axes[row, col].imshow(patch, cmap='gray', vmin=0, vmax=255)
            
            axes[row, col].axis('off')
            
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=14, pad=10)
    
    plt.tight_layout()
    plt.savefig('fig_sar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_val_loss_curve():
    ckpt_path = Path('./results_hybrid_plus_v2_finetuned/checkpoints/stagewise.pt')
    if not ckpt_path.exists():
        print("Checkpoint not found!")
        return
        
    ckpt = torch.load(ckpt_path, map_location='cpu')
    histories = ckpt.get('stage_histories', [])
    
    if not histories:
        print("No stage histories found.")
        return
        
    all_val_loss = []
    stage_boundaries = [0]
    
    for h in histories:
        val_losses = h['history']['val_loss']
        all_val_loss.extend(val_losses)
        stage_boundaries.append(len(all_val_loss))
        
    plt.figure(figsize=(10, 6))
    
    # Plot the curve
    plt.plot(all_val_loss, color='#d32f2f', linewidth=2.5, label='Validation Loss')
    
    # Draw vertical lines for stage boundaries
    for i in range(1, len(stage_boundaries)-1):
        plt.axvline(x=stage_boundaries[i], color='k', linestyle='--', alpha=0.5)
        plt.text(stage_boundaries[i-1] + (stage_boundaries[i] - stage_boundaries[i-1])/2, 
                 max(all_val_loss)*0.95, f'Stage {i}', ha='center', fontsize=12)
                 
    # Label the final stage
    plt.text(stage_boundaries[-2] + (stage_boundaries[-1] - stage_boundaries[-2])/2, 
             max(all_val_loss)*0.95, f'Stage {len(histories)}', ha='center', fontsize=12)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlabel('Training Epochs (Cumulative)', fontsize=14)
    plt.ylabel('MSE Loss', fontsize=14)
    plt.title('Validation Loss Convergence (Stage-wise Training)', fontsize=16, pad=15)
    
    plt.tight_layout()
    plt.savefig('fig_val_loss_curve.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    generate_performance_scatter()
    generate_sar_comparison()
    generate_val_loss_curve()
    print("Generated all figures successfully.")
