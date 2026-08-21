from __future__ import annotations

import numpy as np
import pytest

from aircraft_damage.vision.model import build_feature_extractor, build_model

INPUT_SHAPE = (224, 224, 3)


def test_feature_extractor_is_entirely_frozen() -> None:
    extractor = build_feature_extractor(INPUT_SHAPE, weights=None)

    assert all(not layer.trainable for layer in extractor.layers)


def test_feature_extractor_flattens_to_the_vgg16_feature_vector() -> None:
    extractor = build_feature_extractor(INPUT_SHAPE, weights=None)

    assert extractor.output_shape == (None, 7 * 7 * 512)


def test_model_has_the_notebook_topology() -> None:
    model = build_model(INPUT_SHAPE, learning_rate=0.0001, weights=None)

    layer_types = [type(layer).__name__ for layer in model.layers]
    assert layer_types == [
        "Functional",
        "Dense",
        "Dropout",
        "Dense",
        "Dropout",
        "Dense",
    ]


def test_model_emits_a_single_sigmoid_probability() -> None:
    model = build_model(INPUT_SHAPE, learning_rate=0.0001, weights=None)

    assert model.output_shape == (None, 1)
    assert model.layers[-1].activation.__name__ == "sigmoid"


def test_model_is_compiled_with_the_notebook_optimiser_and_loss() -> None:
    model = build_model(INPUT_SHAPE, learning_rate=0.0001, weights=None)

    assert "adam" in model.optimizer.name.lower()
    assert float(model.optimizer.learning_rate) == pytest.approx(0.0001, rel=1e-5)
    assert "binary_crossentropy" in str(model.loss)


def test_only_the_classifier_head_is_trainable() -> None:
    model = build_model(INPUT_SHAPE, learning_rate=0.0001, weights=None)

    frozen = model.layers[0].count_params()
    assert frozen > 14_000_000
    assert sum(int(np.prod(w.shape)) for w in model.layers[0].trainable_weights) == 0


def test_prediction_is_a_probability_per_image() -> None:
    model = build_model(INPUT_SHAPE, learning_rate=0.0001, weights=None)

    predictions = model.predict(np.zeros((2, *INPUT_SHAPE), dtype=np.float32), verbose=0)

    assert predictions.shape == (2, 1)
    assert np.all((predictions >= 0.0) & (predictions <= 1.0))
