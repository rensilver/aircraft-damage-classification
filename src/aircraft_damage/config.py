"""Project configuration.

This module imports the standard library only. Streamlit reads configuration
before deciding whether to pay TensorFlow's import cost, so keep it that way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "aircraft_damage_dataset_v1"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
MODEL_FILENAME = "vgg16_damage_classifier.keras"
METRICS_FILENAME = "metrics.json"
ACCURACY_CURVE_FILENAME = "accuracy_curve.png"
LOSS_CURVE_FILENAME = "loss_curve.png"


@dataclass(frozen=True)
class Config:
    """Immutable settings for training, inference, and reporting.

    Defaults are copied verbatim from the source notebook so that a training run
    here is comparable to the graded notebook run.
    """

    data_dir: Path
    artifacts_dir: Path
    img_rows: int = 224
    img_cols: int = 224
    batch_size: int = 32
    n_epochs: int = 5
    seed: int = 42
    learning_rate: float = 0.0001
    blip_model_id: str = "Salesforce/blip-image-captioning-base"
    ollama_host: str = DEFAULT_OLLAMA_HOST
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_timeout_s: int = 600

    @property
    def train_dir(self) -> Path:
        """Directory holding the training split."""
        return self.data_dir / "train"

    @property
    def valid_dir(self) -> Path:
        """Directory holding the validation split."""
        return self.data_dir / "valid"

    @property
    def test_dir(self) -> Path:
        """Directory holding the test split."""
        return self.data_dir / "test"

    @property
    def target_size(self) -> tuple[int, int]:
        """Image size, as Keras generators expect it."""
        return (self.img_rows, self.img_cols)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Model input shape, including the channel dimension."""
        return (self.img_rows, self.img_cols, 3)

    @property
    def model_path(self) -> Path:
        """Path of the saved Keras classifier."""
        return self.artifacts_dir / MODEL_FILENAME

    @property
    def metrics_path(self) -> Path:
        """Path of the training metrics JSON."""
        return self.artifacts_dir / METRICS_FILENAME

    @property
    def accuracy_curve_path(self) -> Path:
        """Path of the accuracy curve image."""
        return self.artifacts_dir / ACCURACY_CURVE_FILENAME

    @property
    def loss_curve_path(self) -> Path:
        """Path of the loss curve image."""
        return self.artifacts_dir / LOSS_CURVE_FILENAME


def load_config() -> Config:
    """Build a :class:`Config` from environment variables, falling back to defaults.

    Returns:
        The configuration for this process.
    """
    return Config(
        data_dir=Path(os.environ.get("ADC_DATA_DIR", DEFAULT_DATA_DIR)),
        artifacts_dir=Path(os.environ.get("ADC_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)),
        ollama_host=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        ollama_model=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
    )
