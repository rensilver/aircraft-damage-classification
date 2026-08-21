from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

CLASS_NAMES = ("crack", "dent")
SPLITS = ("train", "valid", "test")
IMAGES_PER_CLASS = 3


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Build a tiny on-disk dataset with the real folder layout.

    Returns:
        Root directory containing ``train/``, ``valid/`` and ``test/`` splits,
        each with ``crack/`` and ``dent/`` subfolders of small RGB JPEGs.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "dataset"
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            folder = root / split / class_name
            folder.mkdir(parents=True)
            for index in range(IMAGES_PER_CLASS):
                pixels = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{class_name}_{index}.jpg")
    return root


@pytest.fixture
def sample_image() -> Image.Image:
    """A single small RGB image, for inference-side tests."""
    rng = np.random.default_rng(1)
    pixels = rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)
    return Image.fromarray(pixels)
