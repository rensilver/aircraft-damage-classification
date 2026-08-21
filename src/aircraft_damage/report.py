"""Turn model outputs into a maintenance report with a local LLM.

The LLM has no vision encoder. Everything it knows about the image arrives as
text in an :class:`EvidencePacket`, and the system prompt says so explicitly so
the report never claims direct observation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from aircraft_damage.llm import OllamaClient

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.2
LOW_CONFIDENCE_THRESHOLD = 0.70

REPORT_SECTIONS = (
    "Finding",
    "Classification",
    "Visual Description",
    "Severity Assessment",
    "Recommended Actions",
    "Limitations",
)

SYSTEM_PROMPT = """You are a senior aircraft structural maintenance inspector \
writing a preliminary damage assessment.

You CANNOT see the image. You are given a structured evidence packet produced by \
an automated pipeline:
- a VGG16 classifier verdict ("crack" or "dent") with a confidence score
- a BLIP image-captioning model's caption and longer description
- basic file metadata

Rules:
1. Reason only from the evidence packet. Never invent details you were not given.
2. Never claim to have viewed the image. Attribute observations to "the classifier" \
and "the image description model".
3. If classifier confidence is below 0.70, treat the classification as provisional \
and say so explicitly.
4. BLIP is a general-purpose captioner and is often vague or wrong about aircraft \
context. Treat its text as weak evidence and say so wherever you rely on it.
5. This is a preliminary triage aid, not an airworthiness determination.

Write the report in Markdown with exactly these sections, in this order:
## Finding
## Classification
## Visual Description
## Severity Assessment
## Recommended Actions
## Limitations

Keep it under 500 words. Be specific and operational. No preamble, no closing \
pleasantries."""


@dataclass(frozen=True)
class EvidencePacket:
    """Everything the LLM is allowed to know about one inspected image.

    Attributes:
        filename: Original name of the uploaded file.
        image_size: Width and height in pixels.
        predicted_label: The classifier's chosen class.
        confidence: Probability assigned to ``predicted_label``.
        probabilities: Probability for every class name.
        caption: BLIP's short caption.
        summary: BLIP's longer description.
        model_test_accuracy: Classifier accuracy on the held-out test split, if known.
    """

    filename: str
    image_size: tuple[int, int]
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    caption: str
    summary: str
    model_test_accuracy: float | None


def build_user_prompt(packet: EvidencePacket) -> str:
    """Render the evidence packet as the LLM's user message.

    Args:
        packet: The evidence gathered for one image.

    Returns:
        A plain-text prompt. Contains no image data of any kind.
    """
    probability_lines = "\n".join(
        f"    - {name}: {probability:.4f}"
        for name, probability in sorted(packet.probabilities.items())
    )
    accuracy = (
        f"{packet.model_test_accuracy:.4f}" if packet.model_test_accuracy is not None else "unknown"
    )
    width, height = packet.image_size

    return f"""EVIDENCE PACKET

File name: {packet.filename}
Image size: {width}x{height} px

Classifier — VGG16 feature extractor with a dense head, binary crack vs dent
    Predicted class: {packet.predicted_label}
    Confidence: {packet.confidence:.4f}
    Low-confidence threshold: {LOW_CONFIDENCE_THRESHOLD:.2f}
    Class probabilities:
{probability_lines}
    Accuracy on the held-out test split: {accuracy}

Image description model — BLIP image-captioning-base
    Caption: {packet.caption}
    Description: {packet.summary}

Write the preliminary damage assessment report."""


def generate_report(
    packet: EvidencePacket,
    client: OllamaClient,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Ask the LLM to write the report for one evidence packet.

    Args:
        packet: The evidence gathered for one image.
        client: A connected Ollama client.
        temperature: Sampling temperature; low keeps the report factual.

    Returns:
        The report as Markdown.

    Raises:
        OllamaError: If the model is unreachable or returns nothing usable.
    """
    logger.info(
        "Generating report for %s (%s @ %.2f)",
        packet.filename,
        packet.predicted_label,
        packet.confidence,
    )
    return client.chat(SYSTEM_PROMPT, build_user_prompt(packet), temperature=temperature)
