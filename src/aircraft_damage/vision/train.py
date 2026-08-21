"""Train the aircraft damage classifier and write artifacts.

Mirrors sections 1.4 to 1.6 of the source notebook. Run with::

    uv run python -m aircraft_damage.vision.train
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

# isort: off
from aircraft_damage import tf_env  # noqa: F401  # must precede the tensorflow import

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from aircraft_damage.config import Config, load_config  # noqa: E402
from aircraft_damage.vision.data import build_generators  # noqa: E402
from aircraft_damage.vision.model import build_model  # noqa: E402
# isort: on

logger = logging.getLogger(__name__)

FIGURE_SIZE = (5, 5)


def set_seeds(seed: int) -> None:
    """Seed every random source the training run touches.

    Args:
        seed: The seed value; 42 in the source notebook.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_curves(
    history: dict[str, list[float]],
    accuracy_path: Path,
    loss_path: Path,
) -> None:
    """Write accuracy and loss curve images.

    Args:
        history: Keras ``History.history`` mapping metric name to per-epoch values.
        accuracy_path: Destination for the accuracy curve.
        loss_path: Destination for the loss curve.
    """
    accuracy_path.parent.mkdir(parents=True, exist_ok=True)

    for title, ylabel, keys, path in (
        ("Accuracy Curve", "Accuracy", ("accuracy", "val_accuracy"), accuracy_path),
        ("Loss Curve", "Loss", ("loss", "val_loss"), loss_path),
    ):
        figure = plt.figure(figsize=FIGURE_SIZE)
        for key in keys:
            if key in history:
                plt.plot(history[key], label=key.replace("_", " ").title())
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend()
        figure.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(figure)

    logger.info("Wrote curves to %s and %s", accuracy_path, loss_path)


def save_metrics(
    path: Path,
    *,
    history: dict[str, list[float]],
    test_loss: float,
    test_accuracy: float,
    class_indices: dict[str, int],
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    """Write the training metrics JSON.

    Keras history values are numpy floats, which ``json`` cannot serialise, so
    everything numeric is coerced to a Python float first.

    Args:
        path: Destination file; parent directories are created.
        history: Keras ``History.history``.
        test_loss: Loss on the held-out test split.
        test_accuracy: Accuracy on the held-out test split.
        class_indices: The generator's class-name-to-index mapping.
        epochs: Number of epochs trained.
        seed: Seed used for the run.

    Returns:
        The dictionary that was written.
    """
    metrics: dict[str, Any] = {
        "history": {key: [float(value) for value in values] for key, values in history.items()},
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "class_indices": dict(class_indices),
        "epochs": int(epochs),
        "seed": int(seed),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote metrics to %s", path)
    return metrics


def run_training(config: Config, *, weights: str | None = "imagenet") -> dict[str, Any]:
    """Run the full training, evaluation, and artifact-writing cycle.

    Args:
        config: Paths and hyperparameters for the run.
        weights: Passed to the feature extractor; ``None`` in fast tests.

    Returns:
        The metrics dictionary, as written to ``config.metrics_path``.
    """
    set_seeds(config.seed)

    train_generator, valid_generator, test_generator = build_generators(
        config.train_dir,
        config.valid_dir,
        config.test_dir,
        target_size=config.target_size,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    model = build_model(
        config.input_shape,
        learning_rate=config.learning_rate,
        weights=weights,
    )

    history = model.fit(
        train_generator,
        epochs=config.n_epochs,
        validation_data=valid_generator,
    )

    # The notebook passes steps=samples//batch_size here, which silently drops the
    # final partial batch. Evaluating the whole split is the correct behaviour.
    test_loss, test_accuracy = model.evaluate(test_generator)
    logger.info("Test loss %.4f, test accuracy %.4f", test_loss, test_accuracy)

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(config.model_path)
    logger.info("Saved model to %s", config.model_path)

    plot_curves(history.history, config.accuracy_curve_path, config.loss_curve_path)

    return save_metrics(
        config.metrics_path,
        history=history.history,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        class_indices=train_generator.class_indices,
        epochs=config.n_epochs,
        seed=config.seed,
    )


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train the aircraft damage classifier.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch count.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override the dataset root.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Override the artifacts output directory.",
    )
    args = parser.parse_args()

    config = load_config()
    if args.epochs is not None:
        config = replace(config, n_epochs=args.epochs)
    if args.data_dir is not None:
        config = replace(config, data_dir=args.data_dir)
    if args.artifacts_dir is not None:
        config = replace(config, artifacts_dir=args.artifacts_dir)

    metrics = run_training(config)

    print(f"Test loss:     {metrics['test_loss']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Artifacts in:  {config.artifacts_dir}")


if __name__ == "__main__":
    main()
