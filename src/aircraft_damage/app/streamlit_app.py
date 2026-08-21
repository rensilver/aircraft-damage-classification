"""Streamlit UI for the aircraft damage inspection pipeline.

Run with::

    uv run streamlit run src/aircraft_damage/app/streamlit_app.py

This module contains no domain logic. Heavy models are imported and loaded lazily
inside cached loaders, because this host has 6 GB of RAM and Ollama already holds
most of it.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import streamlit as st
from PIL import Image

from aircraft_damage.app.styles import CUSTOM_CSS
from aircraft_damage.classifier import DamageClassifier, ModelNotTrainedError
from aircraft_damage.config import Config, load_config
from aircraft_damage.llm import OllamaClient, OllamaError
from aircraft_damage.pipeline import build_packet
from aircraft_damage.report import LOW_CONFIDENCE_THRESHOLD, generate_report

logger = logging.getLogger(__name__)

PAGE_TITLE = "Aircraft Damage Inspection"
UPLOAD_TYPES = ["png", "jpg", "jpeg"]
ANALYSIS_KEY = "analysis_key"
PACKET_KEY = "packet"
REPORT_KEY = "report"


@st.cache_resource(show_spinner=False)
def load_classifier(model_path: Path, metrics_path: Path) -> DamageClassifier:
    """Load the trained classifier once per session.

    Args:
        model_path: Path to the saved ``.keras`` model.
        metrics_path: Path to ``metrics.json``.

    Returns:
        The classifier.
    """
    return DamageClassifier.load(model_path, metrics_path)


@st.cache_resource(show_spinner=False)
def load_describer(model_id: str) -> object:
    """Load BLIP once per session.

    The import lives here so that starting the app without ever uploading an image
    never pays TensorFlow's and torch's import cost.

    Args:
        model_id: Hugging Face model id.

    Returns:
        A ``BlipDescriber``.
    """
    from aircraft_damage.captioning import BlipDescriber  # noqa: PLC0415

    return BlipDescriber.load(model_id)


def read_test_accuracy(metrics_path: Path) -> float | None:
    """Read the classifier's held-out accuracy, if it has been trained.

    Args:
        metrics_path: Path to ``metrics.json``.

    Returns:
        The accuracy, or ``None`` if the file is absent or malformed.
    """
    if not metrics_path.exists():
        return None
    try:
        return float(json.loads(metrics_path.read_text())["test_accuracy"])
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning("Could not read test_accuracy from %s", metrics_path)
        return None


def render_sidebar(config: Config, client: OllamaClient) -> tuple[float, bool]:
    """Draw the health panel and controls.

    Args:
        config: Active configuration.
        client: Ollama client, probed for reachability.

    Returns:
        The chosen temperature and whether the regenerate button was clicked.
    """
    with st.sidebar:
        st.subheader("System status")

        if config.model_path.exists():
            st.success("Classifier artifact found")
        else:
            st.error("No trained classifier")
            st.code("uv run python -m aircraft_damage.train", language="bash")

        if not client.is_available():
            st.error(f"Ollama unreachable at {config.ollama_host}")
            st.code("docker compose up -d", language="bash")
        elif not client.has_model():
            st.error(f"Model {config.ollama_model} not pulled")
            st.code(
                f"docker compose exec ollama ollama pull {config.ollama_model}", language="bash"
            )
        else:
            st.success(f"Ollama ready — {config.ollama_model}")

        st.divider()
        st.subheader("Report settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Lower keeps the report factual; higher makes it more speculative.",
        )
        regenerate = st.button("Regenerate report", use_container_width=True)

        st.divider()
        st.caption(
            f"{config.ollama_model} has no vision encoder. It writes from the "
            "classifier and caption text only — it never sees the image."
        )
        return temperature, regenerate


def render_analysis(packet: object) -> None:
    """Draw the classification metrics and the raw BLIP text.

    Args:
        packet: The :class:`EvidencePacket` for the current image.
    """
    label = packet.predicted_label  # type: ignore[attr-defined]
    confidence = packet.confidence  # type: ignore[attr-defined]
    probabilities = packet.probabilities  # type: ignore[attr-defined]

    pill_class = "adc-pill-crack" if label == "crack" else "adc-pill-dent"
    st.markdown(
        f'<span class="adc-pill {pill_class}">{label}</span>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    left.metric("Predicted damage", label.title())
    right.metric("Confidence", f"{confidence:.1%}")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            f"Confidence is below {LOW_CONFIDENCE_THRESHOLD:.0%}. "
            "Treat this classification as provisional."
        )

    st.bar_chart(probabilities, horizontal=True, height=140)

    with st.expander("Raw model outputs"):
        st.markdown(f"**BLIP caption** — {packet.caption}")  # type: ignore[attr-defined]
        st.markdown(f"**BLIP description** — {packet.summary}")  # type: ignore[attr-defined]


def main() -> None:
    """Render the app."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="✈️", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    config = load_config()
    client = OllamaClient(config.ollama_host, config.ollama_model, config.ollama_timeout_s)
    temperature, regenerate = render_sidebar(config, client)

    st.markdown(
        f"""
        <div class="adc-hero">
          <h1>{PAGE_TITLE}</h1>
          <p>VGG16 classifies the damage, BLIP describes the image, and
             {config.ollama_model} writes the maintenance report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Aircraft surface photo", type=UPLOAD_TYPES)
    if uploaded is None:
        st.info("Upload a JPEG or PNG of an aircraft surface to start an inspection.")
        return

    payload = uploaded.getvalue()
    image = Image.open(io.BytesIO(payload))
    analysis_key = f"{uploaded.name}:{len(payload)}"

    preview, results = st.columns([1, 1.4], gap="large")
    with preview:
        st.image(image, caption=uploaded.name, use_column_width=True)

    with results:
        if st.session_state.get(ANALYSIS_KEY) != analysis_key:
            try:
                with st.status("Inspecting image", expanded=True) as status:
                    st.write("Classifying damage with VGG16…")
                    classifier = load_classifier(config.model_path, config.metrics_path)
                    classification = classifier.predict(image)
                    st.write(
                        f"Classified as **{classification.label}** "
                        f"({classification.confidence:.1%})"
                    )

                    st.write("Describing the image with BLIP…")
                    describer = load_describer(config.blip_model_id)
                    description = describer.describe(image)  # type: ignore[attr-defined]
                    st.write(f"Caption: _{description.caption}_")

                    status.update(label="Analysis complete", state="complete", expanded=False)
            except ModelNotTrainedError as exc:
                st.error(str(exc))
                return
            except OSError as exc:
                st.error(f"Could not analyse this image: {exc}")
                return

            st.session_state[ANALYSIS_KEY] = analysis_key
            st.session_state[PACKET_KEY] = build_packet(
                image,
                uploaded.name,
                classification,
                description,
                read_test_accuracy(config.metrics_path),
            )
            st.session_state.pop(REPORT_KEY, None)

        render_analysis(st.session_state[PACKET_KEY])

    if REPORT_KEY not in st.session_state or regenerate:
        with st.status(f"Writing report with {config.ollama_model}…") as status:
            try:
                st.session_state[REPORT_KEY] = generate_report(
                    st.session_state[PACKET_KEY],
                    client,
                    temperature=temperature,
                )
                status.update(label="Report ready", state="complete")
            except OllamaError as exc:
                status.update(label="Report generation failed", state="error")
                st.error(str(exc))
                return

    st.subheader("Preliminary damage assessment")
    with st.container(border=True):
        st.markdown(st.session_state[REPORT_KEY])
    st.markdown(
        '<p class="adc-note">Preliminary triage aid only. Not an airworthiness '
        "determination. The language model did not see the image.</p>",
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download report",
        data=st.session_state[REPORT_KEY],
        file_name=f"report-{Path(uploaded.name).stem}.md",
        mime="text/markdown",
    )


main()
