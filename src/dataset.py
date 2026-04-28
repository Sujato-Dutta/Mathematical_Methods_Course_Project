from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


@dataclass(frozen=True)
class DatasetSplit:
    train: list[Path]
    val: list[Path]
    test: list[Path]


class RandomAugment180:
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            image = np.fliplr(image)
        if random.random() < 0.5:
            image = np.flipud(image)
        if random.random() < 0.5:
            image = np.rot90(image, 2)
        return np.ascontiguousarray(image)


class GrayscaleImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], augment: bool = False) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.augment = RandomAugment180() if augment else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'Unable to read image: {image_path}')

        image = image.astype(np.float32) / 255.0
        if self.augment is not None:
            image = self.augment(image)
        tensor = torch.from_numpy(image).unsqueeze(0)
        return {'image': tensor, 'name': image_path.stem}


def discover_images(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f'Dataset path does not exist: {root_path}')

    image_paths = sorted(
        path for path in root_path.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f'No supported images found under: {root_path}')
    return image_paths


def split_dataset(
    image_paths: Sequence[Path],
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> DatasetSplit:
    ordered = [Path(path) for path in sorted(image_paths)]
    rng = random.Random(seed)
    rng.shuffle(ordered)
    total = len(ordered)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError('Dataset is too small for an 80/10/10 split.')

    train = ordered[:train_count]
    val = ordered[train_count : train_count + val_count]
    test = ordered[train_count + val_count :]
    return DatasetSplit(train=train, val=val, test=test)


def build_loader(paths: Sequence[Path], batch_size: int, shuffle: bool, augment: bool) -> DataLoader:
    return DataLoader(
        GrayscaleImageDataset(paths, augment=augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
