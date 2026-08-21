"""Image captioning and description with BLIP.

Mirrors Part 2 of the source notebook. Two interfaces are provided:

* :class:`BlipCaptionSummaryLayer` and :func:`generate_text` reproduce the
  notebook's custom Keras layer, which takes image *paths* as string tensors.
* :class:`BlipDescriber` is the plain-Python interface the Streamlit app uses; it
  works on in-memory PIL images, which is what a file upload gives you.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

# isort: off
from aircraft_damage import tf_env  # noqa: F401  # must precede the tensorflow import

import tensorflow as tf  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import BlipForConditionalGeneration, BlipProcessor  # noqa: E402
# isort: on

logger = logging.getLogger(__name__)

CAPTION_PROMPT = "This is a picture of"
SUMMARY_PROMPT = "This is a detailed photo showing"
CAPTION_MAX_NEW_TOKENS = 30
SUMMARY_MAX_NEW_TOKENS = 60
CAPTION_TASK = "caption"


@dataclass(frozen=True)
class ImageDescription:
    """What the captioning model says about an image.

    Attributes:
        caption: Short caption, from ``CAPTION_PROMPT``.
        summary: Longer description, from ``SUMMARY_PROMPT``.
    """

    caption: str
    summary: str


class BlipDescriber:
    """Generates a caption and a description for a PIL image."""

    def __init__(self, processor: Any, model: Any) -> None:
        """Initialise the describer.

        Args:
            processor: A ``BlipProcessor`` or a compatible stub.
            model: A ``BlipForConditionalGeneration`` or a compatible stub.
        """
        self._processor = processor
        self._model = model

    @classmethod
    def load(cls: type[BlipDescriber], model_id: str) -> BlipDescriber:
        """Download (or load from cache) the BLIP processor and model.

        Args:
            model_id: A Hugging Face model id.

        Returns:
            A ready-to-use describer.
        """
        logger.info("Loading BLIP model %s", model_id)
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        return cls(processor, model)

    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Run one conditional generation pass.

        Args:
            image: The image to describe.
            prompt: The conditioning prefix.
            max_new_tokens: Generation budget.

        Returns:
            The decoded text, stripped.
        """
        inputs = self._processor(images=image.convert("RGB"), text=prompt, return_tensors="pt")
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        return str(self._processor.decode(output[0], skip_special_tokens=True)).strip()

    def describe(self, image: Image.Image) -> ImageDescription:
        """Produce both a caption and a longer description.

        Args:
            image: The image to describe.

        Returns:
            The caption and description.
        """
        return ImageDescription(
            caption=self._generate(image, CAPTION_PROMPT, CAPTION_MAX_NEW_TOKENS),
            summary=self._generate(image, SUMMARY_PROMPT, SUMMARY_MAX_NEW_TOKENS),
        )


class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
    """The notebook's custom Keras layer wrapping BLIP.

    Kept for parity with the graded exercise. It accepts an image *path* as a
    string tensor; the Streamlit app uses :class:`BlipDescriber` instead.
    """

    def __init__(self, processor: Any, model: Any, **kwargs: Any) -> None:
        """Initialise the layer.

        Args:
            processor: The BLIP processor.
            model: The BLIP model.
            **kwargs: Forwarded to ``tf.keras.layers.Layer``.
        """
        super().__init__(**kwargs)
        self.processor = processor
        self.model = model

    def call(self, image_path: tf.Tensor, task: tf.Tensor) -> tf.Tensor:
        """Generate text for the image at ``image_path``.

        Args:
            image_path: Scalar string tensor holding a filesystem path.
            task: Scalar string tensor, ``"caption"`` or anything else for a summary.

        Returns:
            A scalar string tensor holding the generated text.
        """
        return tf.py_function(self.process_image, [image_path, task], tf.string)

    def process_image(self, image_path: tf.Tensor, task: tf.Tensor) -> str:
        """Load, preprocess, and describe an image.

        Args:
            image_path: Scalar string tensor holding a filesystem path.
            task: Scalar string tensor selecting the prompt.

        Returns:
            The generated text.

        Raises:
            OSError: If the image cannot be opened.
        """
        image = Image.open(image_path.numpy().decode("utf-8")).convert("RGB")
        is_caption = task.numpy().decode("utf-8") == CAPTION_TASK
        prompt = CAPTION_PROMPT if is_caption else SUMMARY_PROMPT

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        output = self.model.generate(**inputs)
        return str(self.processor.decode(output[0], skip_special_tokens=True))


def generate_text(
    image_path: tf.Tensor,
    task: tf.Tensor,
    processor: Any,
    model: Any,
) -> tf.Tensor:
    """Notebook helper: describe an image through the custom Keras layer.

    Args:
        image_path: Scalar string tensor holding a filesystem path.
        task: Scalar string tensor, ``"caption"`` or ``"summary"``.
        processor: The BLIP processor.
        model: The BLIP model.

    Returns:
        A scalar string tensor holding the generated text.
    """
    blip_layer = BlipCaptionSummaryLayer(processor, model)
    return blip_layer(image_path, task)
