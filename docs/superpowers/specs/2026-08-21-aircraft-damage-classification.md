# Aircraft Damage Classification & Reporting — Spec

**Date:** 2026-08-21
**Source material:** `~/workspaces/ibm-ai-engineering/02_deep_learning_neural_networks_keras/07_Final_Project_Classification_and_Captioning.ipynb`

## Problem

The source notebook is a linear, 87-cell IBM course exercise. It trains a VGG16
feature-extractor classifier to label aircraft damage as `crack` or `dent`, and
uses BLIP (wrapped in a custom Keras layer) to caption and describe images. All
state lives in globals; nothing is reusable, testable, or deployable.

We want that same pipeline restructured as a real Python project, extended with a
local LLM that turns raw model outputs into a readable maintenance report, and
fronted by a Streamlit app that accepts an image upload and returns
classification + report.

## Goals

1. Port the notebook's ML pipeline into tested, importable modules under `src/`.
2. Reproduce the notebook's training run from a CLI, saving versioned artifacts.
3. Add a `qwen3:4b` report-generation stage served by Ollama in Docker.
4. Ship a Streamlit app: upload image -> classify -> describe -> report -> download.

## Non-Goals

- Improving on the notebook's model accuracy (no augmentation, no early stopping,
  no LR schedule). Training must stay faithful to the notebook so results are
  comparable to the graded submission.
- Object detection / damage localisation / bounding boxes.
- Multi-user deployment, authentication, or persistence of past inspections.
- Fine-tuning the LLM.

## Key Architectural Decision: the LLM is text-only

`qwen3:4b` is the **dense** Qwen3 4B model. It has no vision encoder and cannot
see the uploaded image. This is accepted deliberately.

The pipeline therefore has three stages, each producing text that feeds the next:

```
image ──> VGG16 classifier ──> label + confidence + class probabilities ─┐
     └──> BLIP describer   ──> caption + description ────────────────────┤
                                                                          v
                                                         EvidencePacket (pure text)
                                                                          │
                                                                          v
                                                    qwen3:4b via Ollama ──> Markdown report
```

Consequences that the implementation MUST honour:

- The LLM system prompt states explicitly that the model cannot see the image and
  must reason only from the evidence packet.
- The report must attribute observations to "the classifier" / "the image
  description model", never to direct observation.
- BLIP-base is a general-purpose captioner and is frequently vague or wrong about
  aircraft context. The prompt instructs the model to treat captions as weak
  evidence.
- Confidence below 0.70 must be surfaced as provisional in the report.

## Hardware Constraints (drive several design choices)

Target machine: 8 CPU cores, **6 GB RAM total**, no GPU.

- Ollama runs in Docker with `mem_limit: 4g`, `OLLAMA_MAX_LOADED_MODELS=1`,
  `OLLAMA_NUM_PARALLEL=1`.
- Streamlit runs on the host, not in Docker, to avoid a second ~3 GB image and a
  duplicated TF/torch memory footprint.
- The Streamlit app loads the Keras classifier and BLIP lazily behind
  `@st.cache_resource`; neither is imported at module import time.
- Training is CPU-only (`tensorflow-cpu`), 5 epochs over 300 training images.

## Fidelity to the notebook

These values are copied from the notebook and must not drift:

| Item | Value |
|---|---|
| Seed | 42 (`random`, `numpy`, `tf`) |
| Image size | 224 x 224 x 3 |
| Batch size | 32 |
| Epochs | 5 |
| Rescale | `1./255` |
| `class_mode` | `binary` |
| Shuffle | train `True`, valid/test `False` |
| Base | `VGG16(weights='imagenet', include_top=False)`, all layers frozen, `Flatten` on top |
| Head | Dense(512, relu) -> Dropout(0.3) -> Dense(512, relu) -> Dropout(0.3) -> Dense(1, sigmoid) |
| Optimizer | `Adam(learning_rate=0.0001)` |
| Loss | `binary_crossentropy` |
| BLIP model | `Salesforce/blip-image-captioning-base` |
| Caption prompt | `"This is a picture of"` |
| Summary prompt | `"This is a detailed photo showing"` |

The custom `BlipCaptionSummaryLayer` (the notebook's graded Task 8 artifact) is
kept verbatim in `captioning.py`, including its `tf.py_function` path-tensor
interface, so the notebook's structure survives. The Streamlit app does not use
it — it uses a plain `BlipDescriber` class that works on PIL images directly.

## Accepted divergences from the notebook

| Divergence | Reason |
|---|---|
| `transformers==4.44.2` / `torch==2.4.1+cpu` instead of `4.38.2` / `2.2.0+cpu` | The notebook pins match a 2024 Skills Network lab image; those pins no longer resolve cleanly against current `huggingface_hub`. |
| Python 3.12, not 3.13 | `tensorflow-cpu==2.17.1` has no 3.13 wheels. The host default is 3.13, so the venv must pin 3.12 explicitly. |
| `numpy<2` | `tensorflow-cpu==2.17.1` requires it. |
| `pandas` dropped | Imported in the notebook, never used. |
| Test evaluation runs over the whole split | The notebook passes `steps=test_generator.samples // test_generator.batch_size`, which is `50 // 32 == 1` and scores only 32 of the 50 test images. This is a bug, not a design choice. |
| `BlipCaptionSummaryLayer.process_image` no longer catches every exception | The notebook returns the string `"Error processing image"` on any failure, which is indistinguishable from a caption downstream. Errors now propagate and the UI renders them. |
| `BlipDescriber` sets `max_new_tokens` (caption 30, summary 60) | BLIP's default of 20 tokens truncates the "detailed photo showing" prompt mid-sentence. The graded layer keeps library defaults. |
| `pillow==10.4.0` instead of `11.1.0` | `streamlit==1.39.0` requires `pillow<11`; `10.4.0` is the latest release satisfying that cap and fixes several CVEs present in earlier 10.x releases. |

## Dataset

`aircraft_damage_dataset_v1`, 300 train / 96 valid / 50 test images, balanced
across `crack` and `dent`. Available in two ways:

- Already extracted at
  `~/workspaces/ibm-ai-engineering/02_deep_learning_neural_networks_keras/data/aircraft_damage_dataset_v1`
- Downloadable tarball (URL in the notebook), fetched by `scripts/fetch_dataset.py`

Data is gitignored. So are training artifacts.

## Artifacts contract

Training writes into `artifacts/`:

| File | Contents |
|---|---|
| `vgg16_damage_classifier.keras` | The trained Keras model |
| `metrics.json` | `{"history", "test_loss", "test_accuracy", "class_indices", "epochs", "seed"}` |
| `accuracy_curve.png` | Train vs validation accuracy |
| `loss_curve.png` | Train vs validation loss |

`class_indices` is the generator's name -> index mapping (`{"crack": 0, "dent": 1}`).
Inference inverts it; the label ordering is never hardcoded.

## Streamlit app requirements

- Wide layout, custom CSS, dark-friendly.
- Sidebar health panel: classifier artifact present?, Ollama reachable?, is the
  configured model pulled? Each with a clear pass/fail indicator and the exact
  remediation command when failing.
- Sidebar controls: LLM temperature, "Regenerate report" button (re-runs only the
  LLM stage, reusing the cached classification and description).
- Main flow: uploader (png/jpg/jpeg) -> image preview -> `st.status` stages for
  Classify / Describe / Report.
- Results: `st.metric` for label and confidence, a probability bar chart, an
  expander with the raw BLIP caption and description, the report rendered as
  Markdown, and a download button for `report-<filename>.md`.
- Every failure mode (missing artifact, Ollama down, model not pulled) shows an
  actionable `st.error`, never a traceback.

## Success criteria

1. `pytest` passes with no network access and no model downloads.
2. `python -m aircraft_damage.train` produces all four artifacts, test accuracy
   in the same ballpark as the notebook run.
3. `docker compose up -d` + `ollama pull qwen3:4b` yields a reachable model.
4. `streamlit run` -> upload a test-set image -> a report appears containing all
   six required sections, and no sentence claiming the LLM viewed the image.
