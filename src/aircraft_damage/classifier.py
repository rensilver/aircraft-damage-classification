"""Single-image inference over the trained damage classifier.

Deliberately free of TensorFlow at import time: the Keras import lives inside
:meth:`DamageClassifier.load`, so Streamlit and the unit tests can import this
module without paying for TensorFlow.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from aircraft_damage.data import class_names_from_indices

logger = logging.getLogger(__name__)

DEFAULT_TARGET_SIZE = (224, 224)
DECISION_THRESHOLD = 0.5


class ModelNotTrainedError(FileNotFoundError):
    """Raised when the trained classifier artifact cannot be found."""


class PredictsProbabilities(Protocol):
    """The slice of the Keras model interface this module relies on."""

    def predict(self, batch: Any, verbose: int = ...) -> Any:
        """Return a ``(batch, 1)`` array of probabilities."""


@dataclass(frozen=True)
class ClassificationResult:
    """The classifier's verdict for one image.

    Attributes:
        label: The predicted class name.
        confidence: Probability assigned to ``label``, always in ``[0.5, 1.0]``.
        probabilities: Probability for every class name.
    """

    label: str
    confidence: float
    probabilities: dict[str, float]


class DamageClassifier:
    """Wraps a trained Keras model with the notebook's preprocessing."""

    def __init__(
        self,
        model: PredictsProbabilities,
        class_names: dict[int, str],
        target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    ) -> None:
        """Initialise the classifier.

        Args:
            model: Anything exposing Keras' ``predict``.
            class_names: Index-to-name mapping, from ``class_names_from_indices``.
            target_size: Height and width the model expects.
        """
        self._model = model
        self._class_names = class_names
        self._target_size = target_size

    @classmethod
    def load(cls: type[DamageClassifier], model_path: Path, metrics_path: Path) -> DamageClassifier:
        """Load a trained classifier and its class mapping from disk.

        Args:
            model_path: Path to the saved ``.keras`` model.
            metrics_path: Path to ``metrics.json``, read for ``class_indices``.

        Returns:
            A ready-to-use classifier.

        Raises:
            ModelNotTrainedError: If either artifact is missing.
        """
        if not model_path.exists():
            raise ModelNotTrainedError(
                f"No trained model at {model_path}. "
                "Run 'uv run python -m aircraft_damage.train' first."
            )
        if not metrics_path.exists():
            raise ModelNotTrainedError(
                f"No metrics at {metrics_path}. "
                "Run 'uv run python -m aircraft_damage.train' first."
            )

        from aircraft_damage import tf_env  # noqa: F401, PLC0415, I001

        import keras  # noqa: PLC0415, I001

        model = keras.saving.load_model(model_path)
        class_indices: dict[str, int] = json.loads(metrics_path.read_text())["class_indices"]
        logger.info("Loaded classifier from %s with classes %s", model_path, class_indices)
        return cls(model, class_names_from_indices(class_indices))

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize, convert, and rescale an image into a single-item batch.

        Args:
            image: Any PIL image, any mode, any size.

        Returns:
            A ``(1, height, width, 3)`` float32 array with values in ``[0, 1]``.
        """
        height, width = self._target_size
        resized = image.convert("RGB").resize((width, height))  # PIL takes (w, h)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        return array[np.newaxis, ...]

    def predict(self, image: Image.Image) -> ClassificationResult:
        """Classify one image.

        Args:
            image: The image to classify.

        Returns:
            The label, its confidence, and the full probability distribution.
        """
        batch = self.preprocess(image)
        probability = float(self._model.predict(batch, verbose=0)[0][0])

        negative, positive = self._class_names[0], self._class_names[1]
        is_positive = probability > DECISION_THRESHOLD

        return ClassificationResult(
            label=positive if is_positive else negative,
            confidence=probability if is_positive else 1.0 - probability,
            probabilities={negative: 1.0 - probability, positive: probability},
        )
