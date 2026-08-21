from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from aircraft_damage.vision.captioning import (
    CAPTION_PROMPT,
    SUMMARY_PROMPT,
    BlipDescriber,
    ImageDescription,
)


class StubProcessor:
    """Records the prompts it is asked to encode and returns canned decodings."""

    def __init__(self, decoded: list[str]) -> None:
        """Initialise the stub processor.

        Args:
            decoded: List of strings to return from decode() calls.
        """
        self.decoded = decoded
        self.prompts: list[str] = []

    def __call__(self, images: Any, text: str, return_tensors: str) -> dict[str, Any]:
        self.prompts.append(text)
        return {"pixel_values": None}

    def decode(self, tokens: Any, skip_special_tokens: bool) -> str:
        return self.decoded.pop(0)


class StubBlipModel:
    """Returns a fixed token sequence and records generation kwargs."""

    def __init__(self) -> None:
        """Initialise the stub model."""
        self.max_new_tokens: list[int] = []

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.max_new_tokens.append(kwargs["max_new_tokens"])
        return [[1, 2, 3]]


def test_describe_uses_the_notebook_prompts(sample_image: Image.Image) -> None:
    processor = StubProcessor(["a picture of a dent", "a detailed photo of a dent"])
    describer = BlipDescriber(processor, StubBlipModel())

    describer.describe(sample_image)

    assert processor.prompts == [CAPTION_PROMPT, SUMMARY_PROMPT]


def test_describe_returns_caption_and_summary_in_order(sample_image: Image.Image) -> None:
    processor = StubProcessor(["  a cracked panel  ", "a detailed photo of a cracked panel"])
    describer = BlipDescriber(processor, StubBlipModel())

    description = describer.describe(sample_image)

    assert description.caption == "a cracked panel"
    assert description.summary == "a detailed photo of a cracked panel"


def test_summary_is_allowed_more_tokens_than_the_caption(sample_image: Image.Image) -> None:
    model = StubBlipModel()
    describer = BlipDescriber(StubProcessor(["a", "b"]), model)

    describer.describe(sample_image)

    caption_tokens, summary_tokens = model.max_new_tokens
    assert summary_tokens > caption_tokens


def test_description_is_immutable() -> None:
    description = ImageDescription(caption="a", summary="b")

    with pytest.raises(AttributeError):
        description.caption = "c"  # type: ignore[misc]


@pytest.mark.slow
def test_real_blip_produces_non_empty_text(sample_image: Image.Image) -> None:
    describer = BlipDescriber.load("Salesforce/blip-image-captioning-base")

    description = describer.describe(sample_image)

    assert description.caption.strip()
    assert description.summary.strip()
