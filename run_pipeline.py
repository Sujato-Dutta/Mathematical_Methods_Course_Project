from __future__ import annotations

import argparse
from pathlib import Path

from src.ablation import run_ablation_suite
from src.benchmark_models import DnCNN, FFDNetGray
from src.dataset import build_loader, discover_images, split_dataset
from src.evaluate import evaluate_foe_dataset, write_csv, write_summary_text
from src.noise import LOOK_LEVELS
from src.tnrd_model import HybridStagewiseTNRD, StagewiseTNRD
from src.train import TrainingConfig, train_stagewise_model, train_standard_model, tune_pde
from src.utils import ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='FoE-only gamma denoising benchmark pipeline.')
    parser.add_argument('--foe_path', default='./data/FoETrainingSets180', help='Path to FoETrainingSets180 grayscale images.')
    parser.add_argument('--results_dir', default='./results', help='Directory where all outputs are saved.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for deterministic splits and initialization.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = ensure_dir(Path(args.results_dir))
    checkpoints_dir = ensure_dir(results_root / 'checkpoints')
    ensure_dir(results_root / 'figures')
    ensure_dir(results_root / 'denoised_images')

    set_seed(args.seed)
    device = get_device()
    print(f'torch.cuda.is_available() = {device.type == "cuda"}')
    print(f'Using device: {device}')

    foe_paths = discover_images(args.foe_path)
    foe_split = split_dataset(foe_paths, seed=args.seed)
    print(
        f'FoE split | total={len(foe_paths)} train={len(foe_split.train)} '
        f'val={len(foe_split.val)} test={len(foe_split.test)}'
    )

    train_config = TrainingConfig()
    train_loader = build_loader(foe_split.train, batch_size=train_config.batch_size, shuffle=True, augment=True)
    val_loader = build_loader(foe_split.val, batch_size=1, shuffle=False, augment=False)
    test_loader = build_loader(foe_split.test, batch_size=1, shuffle=False, augment=False)

    pde_config = tune_pde(val_loader, device=device)

    dncnn_models: dict[int, DnCNN] = {}
    for looks in LOOK_LEVELS:
        model = DnCNN().to(device)
        train_standard_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            looks_options=(looks,),
            model_name=f'DnCNN L={looks}',
            checkpoint_path=checkpoints_dir / f'dncnn_L{looks}.pth',
            config=train_config,
        )
        dncnn_models[looks] = model

    ffdnet_model = FFDNetGray().to(device)
    train_standard_model(
        model=ffdnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        looks_options=LOOK_LEVELS,
        model_name='FFDNet',
        checkpoint_path=checkpoints_dir / 'ffdnet.pth',
        config=train_config,
    )

    tnrd_models: dict[int, StagewiseTNRD] = {}
    for looks in LOOK_LEVELS:
        model = StagewiseTNRD(
            num_filters=8,
            num_stages=7,
            use_nonlinearity=True,
            use_skip=True,
            learnable_filters=False,
            use_level_embedding=False,
        ).to(device)
        train_stagewise_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            looks=looks,
            model_name=f'TNRD L={looks}',
            checkpoint_path=checkpoints_dir / f'tnrd_L{looks}.pth',
            config=train_config,
        )
        tnrd_models[looks] = model

    hybrid_models: dict[int, HybridStagewiseTNRD] = {}
    for looks in LOOK_LEVELS:
        model = HybridStagewiseTNRD(pde_config=pde_config).to(device)
        train_stagewise_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            looks=looks,
            model_name=f'Hybrid L={looks}',
            checkpoint_path=checkpoints_dir / f'hybrid_L{looks}.pth',
            config=train_config,
        )
        hybrid_models[looks] = model

    metric_rows, summary_rows = evaluate_foe_dataset(
        loader=test_loader,
        device=device,
        pde_config=pde_config,
        dncnn_models=dncnn_models,
        ffdnet_model=ffdnet_model,
        tnrd_models=tnrd_models,
        hybrid_models=hybrid_models,
        output_root=results_root,
    )
    write_csv(
        results_root / 'metric_rows_foe.csv',
        metric_rows,
        fieldnames=('dataset', 'image', 'looks', 'model', 'psnr', 'ssim'),
    )

    ablation_rows = run_ablation_suite(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        checkpoints_dir=checkpoints_dir,
        config=train_config,
        trained_tnrd=tnrd_models,
        results_root=results_root,
    )
    write_csv(
        results_root / 'ablation_table.csv',
        ablation_rows,
        fieldnames=('looks', 'variant', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std'),
    )

    summary = {
        'device': str(device),
        'dataset': 'FoETrainingSets180',
        'pde_config': {
            'alpha': pde_config.alpha,
            'beta': pde_config.beta,
            'delta_t': pde_config.delta_t,
            'num_iterations': pde_config.num_iterations,
            'sigma': pde_config.sigma,
        },
        'results': summary_rows,
        'ablation': ablation_rows,
    }
    save_json(results_root / 'summary.json', summary)
    write_summary_text(results_root=results_root, summary_rows=summary_rows, pde_config=pde_config)
    print(f'All artifacts saved to {results_root.resolve()}')


if __name__ == '__main__':
    main()
