"""End-to-end inspection: classify, describe, then report.

This module owns the ordering and the data flow. The Streamlit app calls the
individual pieces so it can render progress, but the composed
:func:`run_inspection` is the tested reference path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

from aircraft_damage.reporting.llm import OllamaClient
from aircraft_damage.reporting.report import DEFAULT_TEMPERATURE, EvidencePacket, generate_report
from aircraft_damage.vision.classifier import ClassificationResult, DamageClassifier

if TYPE_CHECKING:
    from aircraft_damage.vision.captioning import BlipDescriber, ImageDescription

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InspectionResult:
    """The full output of one inspection.

    Attributes:
        packet: The evidence the LLM was given.
        report: The Markdown report the LLM produced.
    """

    packet: EvidencePacket
    report: str


def build_packet(
    image: Image.Image,
    filename: str,
    classification: ClassificationResult,
    description: ImageDescription,
    test_accuracy: float | None,
) -> EvidencePacket:
    """Assemble the text-only evidence packet for one image.

    Args:
        image: The inspected image, read only for its dimensions.
        filename: Original name of the uploaded file.
        classification: The classifier's verdict.
        description: BLIP's caption and description.
        test_accuracy: Classifier accuracy on the held-out test split, if known.

    Returns:
        The packet handed to the LLM.
    """
    return EvidencePacket(
        filename=filename,
        image_size=image.size,
        predicted_label=classification.label,
        confidence=classification.confidence,
        probabilities=classification.probabilities,
        caption=description.caption,
        summary=description.summary,
        model_test_accuracy=test_accuracy,
    )


def run_inspection(
    image: Image.Image,
    filename: str,
    *,
    classifier: DamageClassifier,
    describer: BlipDescriber,
    client: OllamaClient,
    test_accuracy: float | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> InspectionResult:
    """Classify, describe, and report on one image.

    Args:
        image: The image to inspect.
        filename: Original name of the uploaded file.
        classifier: The trained damage classifier.
        describer: The BLIP describer.
        client: A connected Ollama client.
        test_accuracy: Classifier accuracy on the held-out test split, if known.
        temperature: Sampling temperature for the report.

    Returns:
        The evidence packet and the generated report.

    Raises:
        OllamaError: If report generation fails.
    """
    classification = classifier.predict(image)
    description = describer.describe(image)
    packet = build_packet(image, filename, classification, description, test_accuracy)
    report = generate_report(packet, client, temperature=temperature)
    return InspectionResult(packet=packet, report=report)
