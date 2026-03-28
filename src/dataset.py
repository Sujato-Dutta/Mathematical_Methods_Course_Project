from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSplit:
    train: list[Path]
    val: list[Path]
    test: list[Path]


class GrayscaleImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path]) -> None:
        self.image_paths = [Path(path) for path in image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0)
        return {"image": tensor, "name": image_path.stem}


def discover_images(root: str | Path) -> list[Path]:
    root_path = Path(root)
    image_paths = sorted(
        path for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {root_path}")
    return image_paths


def split_dataset(
    image_paths: Sequence[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> DatasetSplit:
    image_paths = sorted(Path(path) for path in image_paths)
    total = len(image_paths)
    if total < 10:
        raise ValueError("Expected at least 10 images for train/val/test split.")

    train_end = max(1, int(total * train_ratio))
    val_end = min(total, train_end + max(1, int(total * val_ratio)))

    train_paths = list(image_paths[:train_end])
    val_paths = list(image_paths[train_end:val_end])
    test_paths = list(image_paths[val_end:])

    if not val_paths:
        val_paths = train_paths[-1:]
        train_paths = train_paths[:-1]
    if not test_paths:
        test_paths = val_paths[-1:]
        val_paths = val_paths[:-1]

    return DatasetSplit(train=train_paths, val=val_paths, test=test_paths)


def discover_set12_images(root: str | Path) -> list[Path]:
    return discover_images(root)


def split_set12(image_paths: Sequence[Path]) -> DatasetSplit:
    return split_dataset(image_paths)
