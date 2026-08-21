from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aircraft_damage.train import plot_curves, run_training, save_metrics, set_seeds

HISTORY = {
    "loss": [0.7, 0.5, 0.3],
    "val_loss": [0.8, 0.6, 0.5],
    "accuracy": [0.5, 0.7, 0.9],
    "val_accuracy": [0.4, 0.6, 0.8],
}


def test_set_seeds_makes_numpy_reproducible() -> None:
    set_seeds(42)
    first = np.random.rand(5)
    set_seeds(42)
    second = np.random.rand(5)

    assert np.array_equal(first, second)


def test_plot_curves_writes_both_images(tmp_path: Path) -> None:
    accuracy_path = tmp_path / "accuracy_curve.png"
    loss_path = tmp_path / "loss_curve.png"

    plot_curves(HISTORY, accuracy_path, loss_path)

    assert accuracy_path.stat().st_size > 0
    assert loss_path.stat().st_size > 0


def test_save_metrics_writes_every_required_key(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "metrics.json"

    save_metrics(
        path,
        history=HISTORY,
        test_loss=0.31,
        test_accuracy=0.88,
        class_indices={"crack": 0, "dent": 1},
        epochs=5,
        seed=42,
    )

    written = json.loads(path.read_text())
    assert set(written) == {
        "history",
        "test_loss",
        "test_accuracy",
        "class_indices",
        "epochs",
        "seed",
    }
    assert written["class_indices"] == {"crack": 0, "dent": 1}
    assert written["test_accuracy"] == pytest.approx(0.88)


def test_save_metrics_coerces_numpy_floats_to_json_safe_values(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"

    save_metrics(
        path,
        history={"loss": [np.float32(0.5), np.float32(0.25)]},  # type: ignore[list-item]
        test_loss=np.float32(0.31),  # type: ignore[arg-type]
        test_accuracy=np.float32(0.88),  # type: ignore[arg-type]
        class_indices={"crack": 0, "dent": 1},
        epochs=5,
        seed=42,
    )

    written = json.loads(path.read_text())
    assert written["history"]["loss"] == [0.5, 0.25]
    assert isinstance(written["test_loss"], float)


@pytest.mark.slow
def test_run_training_produces_every_artifact(synthetic_dataset: Path, tmp_path: Path) -> None:
    from dataclasses import replace

    from aircraft_damage.config import Config

    config = replace(
        Config(data_dir=synthetic_dataset, artifacts_dir=tmp_path / "artifacts"),
        n_epochs=1,
        batch_size=2,
    )

    metrics = run_training(config, weights=None)

    assert config.model_path.exists()
    assert config.metrics_path.exists()
    assert config.accuracy_curve_path.exists()
    assert config.loss_curve_path.exists()
    assert metrics["class_indices"] == {"crack": 0, "dent": 1}
