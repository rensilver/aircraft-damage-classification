"""Keras image generators for the aircraft damage dataset.

Mirrors sections 1.1 and 1.2 of the source notebook.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

# isort: off
from aircraft_damage import tf_env  # noqa: F401  # must precede the tensorflow import

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # noqa: E402
# isort: on

if TYPE_CHECKING:
    from tensorflow.keras.preprocessing.image import DirectoryIterator

logger = logging.getLogger(__name__)

RESCALE_FACTOR = 1.0 / 255
CLASS_MODE = "binary"


def build_generators(
    train_dir: Path,
    valid_dir: Path,
    test_dir: Path,
    *,
    target_size: tuple[int, int],
    batch_size: int,
    seed: int,
) -> tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
    """Create the train, validation, and test generators.

    Only the training generator shuffles; validation and test keep file order so
    that predictions line up with ``generator.classes``.

    Args:
        train_dir: Directory containing the training split.
        valid_dir: Directory containing the validation split.
        test_dir: Directory containing the test split.
        target_size: Height and width every image is resized to.
        batch_size: Images per batch.
        seed: Shuffle seed, for reproducibility.

    Returns:
        The train, validation, and test generators, in that order.
    """
    datagen = ImageDataGenerator(rescale=RESCALE_FACTOR)

    def flow(directory: Path, *, shuffle: bool) -> DirectoryIterator:
        return datagen.flow_from_directory(
            directory=str(directory),
            class_mode=CLASS_MODE,
            seed=seed,
            batch_size=batch_size,
            shuffle=shuffle,
            target_size=target_size,
        )

    train_generator = flow(train_dir, shuffle=True)
    valid_generator = flow(valid_dir, shuffle=False)
    test_generator = flow(test_dir, shuffle=False)

    logger.info(
        "Loaded %d train / %d valid / %d test images across classes %s",
        train_generator.samples,
        valid_generator.samples,
        test_generator.samples,
        train_generator.class_indices,
    )
    return train_generator, valid_generator, test_generator


def class_names_from_indices(class_indices: dict[str, int]) -> dict[int, str]:
    """Invert a Keras ``class_indices`` mapping.

    Args:
        class_indices: Mapping of class name to index, as Keras produces it.

    Returns:
        Mapping of index to class name.
    """
    return {index: name for name, index in class_indices.items()}
