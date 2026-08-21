from __future__ import annotations

from aircraft_damage.reporting.report import (
    REPORT_SECTIONS,
    SYSTEM_PROMPT,
    EvidencePacket,
    build_user_prompt,
    generate_report,
)

PACKET = EvidencePacket(
    filename="149_22.jpg",
    image_size=(640, 480),
    predicted_label="dent",
    confidence=0.8734,
    probabilities={"crack": 0.1266, "dent": 0.8734},
    caption="a close up of a metal surface with a dent",
    summary="a detailed photo showing a dented panel on an aircraft wing",
    model_test_accuracy=0.88,
)


class RecordingClient:
    """Captures the prompts it is given and returns a canned report."""

    def __init__(self, reply: str = "## Finding\nA dent.") -> None:
        """Initialise with a canned reply to return from ``chat``.

        Args:
            reply: The text ``chat`` should return.
        """
        self.reply = reply
        self.system: str | None = None
        self.user: str | None = None
        self.temperature: float | None = None

    def chat(self, system: str, user: str, *, temperature: float = 0.2) -> str:
        self.system = system
        self.user = user
        self.temperature = temperature
        return self.reply


def test_system_prompt_states_the_model_cannot_see_the_image() -> None:
    assert "CANNOT see the image" in SYSTEM_PROMPT


def test_system_prompt_names_every_required_section() -> None:
    for section in REPORT_SECTIONS:
        assert f"## {section}" in SYSTEM_PROMPT


def test_user_prompt_carries_the_classifier_verdict() -> None:
    prompt = build_user_prompt(PACKET)

    assert "dent" in prompt
    assert "0.8734" in prompt


def test_user_prompt_carries_every_class_probability() -> None:
    prompt = build_user_prompt(PACKET)

    assert "crack: 0.1266" in prompt
    assert "dent: 0.8734" in prompt


def test_user_prompt_carries_the_blip_text() -> None:
    prompt = build_user_prompt(PACKET)

    assert PACKET.caption in prompt
    assert PACKET.summary in prompt


def test_user_prompt_carries_the_file_metadata() -> None:
    prompt = build_user_prompt(PACKET)

    assert "149_22.jpg" in prompt
    assert "640x480" in prompt


def test_user_prompt_says_unknown_when_test_accuracy_is_missing() -> None:
    from dataclasses import replace

    prompt = build_user_prompt(replace(PACKET, model_test_accuracy=None))

    assert "unknown" in prompt


def test_generate_report_passes_the_system_prompt_through() -> None:
    client = RecordingClient()

    generate_report(PACKET, client)  # type: ignore[arg-type]

    assert client.system == SYSTEM_PROMPT
    assert client.user == build_user_prompt(PACKET)


def test_generate_report_forwards_the_temperature() -> None:
    client = RecordingClient()

    generate_report(PACKET, client, temperature=0.7)  # type: ignore[arg-type]

    assert client.temperature == 0.7


def test_generate_report_returns_the_model_text() -> None:
    client = RecordingClient(reply="## Finding\nA crack near the rivet line.")

    report = generate_report(PACKET, client)  # type: ignore[arg-type]

    assert report == "## Finding\nA crack near the rivet line."
