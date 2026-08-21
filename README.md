# Aircraft Damage Classification & Reporting

Classify aircraft surface damage as **crack** or **dent** with a VGG16 feature
extractor, describe the image with BLIP, and have a local `qwen3:4b` model write a
preliminary maintenance report. Ships with a Streamlit app.

Refactored from the IBM *Deep Learning with Keras* final project notebook.

## How it works

```
image ──> VGG16 classifier ──> label + confidence ─┐
     └──> BLIP describer   ──> caption + summary ──┤
                                                    v
                                        EvidencePacket (text only)
                                                    │
                                                    v
                                   qwen3:4b (Ollama) ──> Markdown report
```

`qwen3:4b` has **no vision encoder**. It never receives image pixels — it writes
the report from the classifier's verdict and BLIP's text alone, and its system
prompt requires it to say so.

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
uv venv --python 3.12
uv sync
```

Get the dataset (300 train / 96 valid / 50 test images):

```bash
uv run python scripts/fetch_dataset.py
```

Start the LLM and pull the model (~2.6 GB):

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen3:4b
```

The first inspection also downloads the BLIP captioning model (~1 GB) from
Hugging Face on first use. If there is no network access on that first run,
the app will show an error until BLIP is cached locally.

## Train

```bash
uv run python -m aircraft_damage.vision.train
```

Writes `artifacts/vgg16_damage_classifier.keras`, `metrics.json`, and two curve
PNGs. Five epochs on CPU takes roughly 10–20 minutes.

## Run the app

```bash
uv run streamlit run src/aircraft_damage/app/streamlit_app.py
```

## Development

```bash
./scripts/check.sh                  # ruff + mypy + tests
uv run pytest -m "not slow"         # fast tests only
uv run pytest -m slow               # downloads BLIP, trains a tiny model
```

Conventions live in `CLAUDE.md`. The design rationale, including every deliberate
divergence from the source notebook, lives in
`docs/superpowers/specs/2026-08-21-aircraft-damage-classification.md`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModelNotTrainedError` | Run the training command above. |
| Sidebar: "Ollama unreachable" | `docker compose up -d`, then wait for the healthcheck. |
| Sidebar: "Model not pulled" | `docker compose exec ollama ollama pull qwen3:4b` |
| First report is slow | Report generation can take several minutes on CPU-only hardware — the client timeout is set to 600 seconds (10 minutes) to accommodate this. Ollama is also loading the model into RAM on the first call; later calls reuse it for 5 minutes. |
| Training killed by the OOM reaper | Close other apps; this host has 6 GB total and Ollama holds up to 4 GB. `docker compose stop` while training. |
| `tensorflow-cpu` will not install | The venv is not Python 3.12. Recreate it with `uv venv --python 3.12`. |

## Limitations

- Two classes only; an undamaged surface is still forced into `crack` or `dent`.
- Trained on 300 images. Treat confidence numbers as indicative, not calibrated.
- BLIP-base is a general-purpose captioner with no aircraft-specific training.
- Output is a triage aid, never an airworthiness determination.
