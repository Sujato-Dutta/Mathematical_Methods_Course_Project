from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_image(path: str | Path, image_tensor: torch.Tensor) -> None:
    image = image_tensor.detach().cpu().squeeze().clamp(0.0, 1.0).numpy()
    image_uint8 = np.round(image * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), image_uint8)


def tensor_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    return image_tensor.detach().cpu().squeeze().clamp(0.0, 1.0).numpy()
