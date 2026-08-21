"""VGG16 feature extractor plus a binary classification head.

Mirrors section 1.3 of the source notebook: the ImageNet-pretrained VGG16
convolutional stack is frozen and used purely as a feature extractor, and only
the dense head on top is trained.
"""

from __future__ import annotations

import logging

import numpy as np

# isort: off
from aircraft_damage import tf_env  # noqa: F401  # must precede the keras import

from keras.applications import VGG16  # noqa: E402
from keras.layers import Dense, Dropout, Flatten  # noqa: E402
from keras.models import Model, Sequential  # noqa: E402
from keras.optimizers import Adam  # noqa: E402
# isort: on

logger = logging.getLogger(__name__)

DENSE_UNITS = 512
DROPOUT_RATE = 0.3
LOSS = "binary_crossentropy"
METRICS = ("accuracy",)


def build_feature_extractor(
    input_shape: tuple[int, int, int],
    *,
    weights: str | None = "imagenet",
) -> Model:
    """Build a frozen VGG16 convolutional stack with a flattened output.

    Args:
        input_shape: Height, width, and channel count of the input images.
        weights: Weight initialisation. ``"imagenet"`` in production; pass
            ``None`` in tests to avoid downloading the pretrained weights.

    Returns:
        A non-trainable model mapping images to a flat feature vector.
    """
    base_model = VGG16(weights=weights, include_top=False, input_shape=input_shape)
    flattened = Flatten()(base_model.layers[-1].output)
    extractor = Model(base_model.input, flattened)
    for layer in extractor.layers:
        layer.trainable = False
    return extractor


def build_model(
    input_shape: tuple[int, int, int],
    *,
    learning_rate: float,
    weights: str | None = "imagenet",
) -> Sequential:
    """Build and compile the crack-vs-dent classifier.

    Args:
        input_shape: Height, width, and channel count of the input images.
        learning_rate: Adam learning rate.
        weights: Passed through to :func:`build_feature_extractor`.

    Returns:
        A compiled model whose single sigmoid output is the probability of the
        class with index 1 in the generator's ``class_indices``.
    """
    model = Sequential(
        [
            build_feature_extractor(input_shape, weights=weights),
            Dense(DENSE_UNITS, activation="relu"),
            Dropout(DROPOUT_RATE),
            Dense(DENSE_UNITS, activation="relu"),
            Dropout(DROPOUT_RATE),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=LOSS,
        metrics=list(METRICS),
    )
    trainable_params = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    logger.info("Built classifier with %d trainable parameters", trainable_params)
    return model
