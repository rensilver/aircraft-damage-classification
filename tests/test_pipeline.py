from __future__ import annotations

from PIL import Image

from aircraft_damage.pipeline import build_packet, run_inspection
from aircraft_damage.vision.captioning import ImageDescription
from aircraft_damage.vision.classifier import ClassificationResult

CLASSIFICATION = ClassificationResult(
    label="dent",
    confidence=0.9,
    probabilities={"crack": 0.1, "dent": 0.9},
)
DESCRIPTION = ImageDescription(caption="a dented panel", summary="a detailed photo of a dent")


class StubClassifier:
    def predict(self, image: Image.Image) -> ClassificationResult:
        return CLASSIFICATION


class StubDescriber:
    def describe(self, image: Image.Image) -> ImageDescription:
        return DESCRIPTION


class StubClient:
    def __init__(self) -> None:
        """Initialize the stub client."""
        self.calls = 0

    def chat(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        self.calls += 1
        return "## Finding\nA dent."


def test_build_packet_records_the_image_dimensions(sample_image: Image.Image) -> None:
    packet = build_packet(sample_image, "x.jpg", CLASSIFICATION, DESCRIPTION, None)

    assert packet.image_size == sample_image.size


def test_build_packet_carries_classification_and_description(
    sample_image: Image.Image,
) -> None:
    packet = build_packet(sample_image, "x.jpg", CLASSIFICATION, DESCRIPTION, 0.88)

    assert packet.filename == "x.jpg"
    assert packet.predicted_label == "dent"
    assert packet.confidence == 0.9
    assert packet.probabilities == {"crack": 0.1, "dent": 0.9}
    assert packet.caption == "a dented panel"
    assert packet.summary == "a detailed photo of a dent"
    assert packet.model_test_accuracy == 0.88


def test_run_inspection_returns_packet_and_report(sample_image: Image.Image) -> None:
    result = run_inspection(
        sample_image,
        "x.jpg",
        classifier=StubClassifier(),  # type: ignore[arg-type]
        describer=StubDescriber(),  # type: ignore[arg-type]
        client=StubClient(),  # type: ignore[arg-type]
        test_accuracy=0.88,
    )

    assert result.packet.predicted_label == "dent"
    assert result.report == "## Finding\nA dent."


def test_run_inspection_calls_the_llm_exactly_once(sample_image: Image.Image) -> None:
    client = StubClient()

    run_inspection(
        sample_image,
        "x.jpg",
        classifier=StubClassifier(),  # type: ignore[arg-type]
        describer=StubDescriber(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
    )

    assert client.calls == 1
