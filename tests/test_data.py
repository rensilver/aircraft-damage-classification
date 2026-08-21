from __future__ import annotations

from pathlib import Path

from aircraft_damage.vision.data import build_generators


def test_generators_discover_both_classes_in_alphabetical_order(
    synthetic_dataset: Path,
) -> None:
    train, valid, test = build_generators(
        synthetic_dataset / "train",
        synthetic_dataset / "valid",
        synthetic_dataset / "test",
        target_size=(224, 224),
        batch_size=4,
        seed=42,
    )

    assert train.class_indices == {"crack": 0, "dent": 1}
    assert valid.class_indices == {"crack": 0, "dent": 1}
    assert test.class_indices == {"crack": 0, "dent": 1}


def test_generators_find_every_image(synthetic_dataset: Path) -> None:
    train, valid, test = build_generators(
        synthetic_dataset / "train",
        synthetic_dataset / "valid",
        synthetic_dataset / "test",
        target_size=(224, 224),
        batch_size=4,
        seed=42,
    )

    assert (train.samples, valid.samples, test.samples) == (6, 6, 6)


def test_batches_are_resized_and_rescaled(synthetic_dataset: Path) -> None:
    train, _, _ = build_generators(
        synthetic_dataset / "train",
        synthetic_dataset / "valid",
        synthetic_dataset / "test",
        target_size=(224, 224),
        batch_size=4,
        seed=42,
    )

    images, labels = next(train)

    assert images.shape[1:] == (224, 224, 3)
    assert images.max() <= 1.0
    assert set(labels.tolist()) <= {0.0, 1.0}


def test_only_the_training_generator_shuffles(synthetic_dataset: Path) -> None:
    train, valid, test = build_generators(
        synthetic_dataset / "train",
        synthetic_dataset / "valid",
        synthetic_dataset / "test",
        target_size=(224, 224),
        batch_size=4,
        seed=42,
    )

    assert train.shuffle is True
    assert valid.shuffle is False
    assert test.shuffle is False
