from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from aircraft_damage.vision.classifier import (
    ClassificationResult,
    DamageClassifier,
    ModelNotTrainedError,
    class_names_from_indices,
)

CLASS_NAMES = {0: "crack", 1: "dent"}


class StubModel:
    """A stand-in for a Keras model that always returns a fixed probability."""

    def __init__(self, probability: float) -> None:
        """Initialise with a fixed probability to return.

        Args:
            probability: The probability value to return on predict calls.
        """
        self.probability = probability
        self.last_batch: Any = None

    def predict(self, batch: Any, verbose: int = 0) -> Any:
        self.last_batch = batch
        return np.array([[self.probability]], dtype=np.float32)


def test_preprocess_produces_one_normalised_224_square(sample_image: Image.Image) -> None:
    classifier = DamageClassifier(StubModel(0.5), CLASS_NAMES)

    batch = classifier.preprocess(sample_image)

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert batch.max() <= 1.0
    assert batch.min() >= 0.0


def test_preprocess_converts_greyscale_to_rgb() -> None:
    classifier = DamageClassifier(StubModel(0.5), CLASS_NAMES)
    grey = Image.new("L", (30, 40), color=128)

    batch = classifier.preprocess(grey)

    assert batch.shape == (1, 224, 224, 3)


def test_high_probability_selects_the_class_at_index_one(sample_image: Image.Image) -> None:
    classifier = DamageClassifier(StubModel(0.87), CLASS_NAMES)

    result = classifier.predict(sample_image)

    assert result.label == "dent"
    assert result.confidence == pytest.approx(0.87)


def test_low_probability_selects_the_class_at_index_zero(sample_image: Image.Image) -> None:
    classifier = DamageClassifier(StubModel(0.12), CLASS_NAMES)

    result = classifier.predict(sample_image)

    assert result.label == "crack"
    assert result.confidence == pytest.approx(0.88)


def test_probabilities_cover_both_classes_and_sum_to_one(sample_image: Image.Image) -> None:
    classifier = DamageClassifier(StubModel(0.3), CLASS_NAMES)

    result = classifier.predict(sample_image)

    assert set(result.probabilities) == {"crack", "dent"}
    assert result.probabilities["dent"] == pytest.approx(0.3)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_result_is_immutable(sample_image: Image.Image) -> None:
    result = ClassificationResult(label="dent", confidence=0.9, probabilities={"dent": 0.9})

    with pytest.raises(AttributeError):
        result.label = "crack"  # type: ignore[misc]


def test_load_raises_a_clear_error_when_the_model_is_missing(tmp_path: Path) -> None:
    with pytest.raises(ModelNotTrainedError, match="train"):
        DamageClassifier.load(tmp_path / "missing.keras", tmp_path / "metrics.json")


def test_class_names_from_indices_inverts_the_mapping() -> None:
    assert class_names_from_indices({"crack": 0, "dent": 1}) == {0: "crack", 1: "dent"}
