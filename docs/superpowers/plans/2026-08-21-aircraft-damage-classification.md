# Aircraft Damage Classification & Reporting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the IBM course notebook into a tested Python project that classifies aircraft damage (crack/dent) with VGG16, describes it with BLIP, writes a maintenance report with a local `qwen3:4b` model, and serves the whole thing through a Streamlit app.

**Architecture:** Three text-producing stages feed one report generator. VGG16 (frozen ImageNet features + a 512/512/1 dense head) yields a label and confidence; BLIP-base yields a caption and a description; both are packed into a plain-text `EvidencePacket` that `qwen3:4b` — running in Docker via Ollama, with no vision encoder — turns into Markdown. All logic lives in importable, unit-tested modules; Streamlit is a thin UI shell over `pipeline.run_inspection`.

**Tech Stack:** Python 3.12, `tensorflow-cpu` 2.17.1 / Keras, `transformers` 4.44.2 + `torch` 2.4.1+cpu (BLIP), `httpx` (Ollama HTTP), Streamlit 1.39, `uv` for env + deps, Docker Compose for Ollama, `pytest` + `ruff` + `mypy`.

**Spec:** `docs/superpowers/specs/2026-08-21-aircraft-damage-classification.md`

---

## Global Constraints

Every task's requirements implicitly include this section.

### Environment & versions

- Project root: `/home/rensilver/workspaces/aircraft-damage-classification`. All paths below are relative to it.
- Python **3.12** exactly. The host default is 3.13, which has no `tensorflow-cpu==2.17.1` wheels. Create the venv with `uv venv --python 3.12`.
- Dependency pins (exact, no ranges except where stated):
  `tensorflow-cpu==2.17.1`, `numpy>=1.26,<2`, `pillow==11.1.0`, `matplotlib==3.9.2`,
  `transformers==4.44.2`, `torch==2.4.1+cpu`, `httpx==0.27.2`, `streamlit==1.39.0`.
  Dev: `pytest==8.3.3`, `ruff==0.6.9`, `mypy==1.11.2`.
- `torch==2.4.1+cpu` resolves only from `https://download.pytorch.org/whl/cpu`; it needs an explicit `uv` index entry (given in Task 1).
- Ollama model id: `qwen3:4b`. Ollama base URL: `http://localhost:11434`.
- Hardware: 8 cores, 6 GB RAM, **no GPU**. Never load two large models eagerly in the same process.

### Values copied verbatim from the notebook — do not change

`seed = 42` · `img_rows, img_cols = 224, 224` · `batch_size = 32` · `n_epochs = 5` ·
`rescale=1./255` · `class_mode='binary'` · train `shuffle=True`, valid/test `shuffle=False` ·
`VGG16(weights='imagenet', include_top=False)` fully frozen with `Flatten` on top ·
head `Dense(512,'relu') -> Dropout(0.3) -> Dense(512,'relu') -> Dropout(0.3) -> Dense(1,'sigmoid')` ·
`Adam(learning_rate=0.0001)` · `loss='binary_crossentropy'` · `metrics=['accuracy']` ·
BLIP model `Salesforce/blip-image-captioning-base` · caption prompt `"This is a picture of"` ·
summary prompt `"This is a detailed photo showing"`.

### Python coding conventions (enforced, not aspirational)

These are checked mechanically. Task 1 wires the tooling; **every subsequent task's
commit step runs `./scripts/check.sh` and must see it pass before committing.**

1. **Type hints on every function and method**, including tests. `from __future__ import annotations` at the top of every module so annotations stay lazy and cheap.
2. **`pathlib.Path`, never `os.path`.** Never build paths with string concatenation. (`ruff` rule set `PTH` enforces this.)
3. **Frozen dataclasses for data.** Anything that is a bag of values is a `@dataclass(frozen=True)`. No dicts-as-records crossing module boundaries, no mutable default arguments.
4. **Pure functions separated from I/O.** Anything that touches disk, network, or a model is a thin wrapper around a pure function that a unit test can call directly. This is why `report.build_user_prompt`, `llm.strip_thinking`, and `classifier.DamageClassifier.predict` are all testable without TensorFlow, torch, or a network.
5. **Dependency injection over globals.** Modules take their collaborators as constructor arguments (`DamageClassifier(model, class_names)`, `run_inspection(..., client=client)`). No module-level model instances, no import-time side effects beyond setting TF env vars.
6. **No bare `except:` and no `except Exception` that swallows.** Catch the narrowest exception type. If you must catch broadly, re-raise as a domain exception (`OllamaError`) with the original chained via `from exc`.
7. **Errors are typed.** Each module that can fail for domain reasons defines its own exception (`OllamaError`). The Streamlit layer catches those and renders `st.error`; it never shows a traceback.
8. **`logging`, never `print`,** in library code under `src/aircraft_damage/`. CLI entry points (`train.py`'s `main`) may print user-facing progress. Get the logger with `logger = logging.getLogger(__name__)` at module level.
9. **Google-style docstrings** on every public module, class, and function. One-line summary, then `Args:` / `Returns:` / `Raises:` when non-obvious. Private helpers (`_leading_underscore`) may have a one-liner.
10. **Module-level constants are `UPPER_SNAKE_CASE`** and live at the top of the file, right after imports. Prompts, regexes, and magic numbers must be named constants — no string literals buried in function bodies.
11. **Keep modules small and single-purpose.** If a module in `src/aircraft_damage/` passes ~150 lines, that is a signal it is doing two jobs. The file structure below is already decomposed this way; do not merge modules to save files.
12. **Tests: arrange / act / assert, one behaviour per test.** Test names read as sentences (`test_predict_returns_lower_class_when_probability_below_half`). No network, no model downloads, and no real training in the default test run — anything that needs those is marked `@pytest.mark.slow` and excluded by `-m "not slow"`.
13. **Fakes over mocks.** Prefer a small hand-written stub class with the right shape to `unittest.mock.MagicMock`; a stub fails loudly when the interface changes, a mock silently accepts anything.
14. **Line length 100.** `ruff format` is the only formatter; never hand-format around it.
15. **Commit messages use Conventional Commits** (`feat:`, `test:`, `chore:`, `docs:`, `fix:`). Commit at the end of every task, never mid-task with a red test suite.

### Git workflow — one branch per task

Every task in this plan is a "major step" and gets **its own branch**, merged back
into `main` only once `./scripts/check.sh` passes. Never commit directly to `main`.

The pattern, repeated for every task (`NN` is the zero-padded task number, `slug`
is the short name given in that task's branch step):

```bash
# at the start of the task
git checkout main
git checkout -b task-NN-slug

# ... implement the task's steps ...

./scripts/check.sh          # must print "All checks passed."
git add -A
git commit -m "<conventional commit message from the task>"

# at the end of the task
git checkout main
git merge --no-ff task-NN-slug -m "merge: task NN <short description>"
```

Rules:

- `--no-ff` is mandatory. Each task must remain a visible, revertable unit in the
  history; a fast-forward merge erases the task boundary.
- One commit per task is the default. If a task genuinely needs intermediate
  commits, each must leave the suite green.
- Do not delete task branches. They are the audit trail of the plan's execution.
- If `./scripts/check.sh` fails, fix it on the task branch. Never merge red.
- The branch names are fixed by this plan, in order:
  `task-01-scaffold`, `task-02-data`, `task-03-model`, `task-04-train`,
  `task-05-classifier`, `task-06-captioning`, `task-07-ollama`, `task-08-report`,
  `task-09-pipeline`, `task-10-app`.

### The text-only LLM rule

`qwen3:4b` has no vision encoder. It never receives pixels. Any code, prompt, or UI
copy implying the LLM looked at the image is a bug. The system prompt in
`report.py` states this constraint explicitly and must not be softened.

---

## File Structure

```
aircraft-damage-classification/
├── CLAUDE.md                     # conventions above, for future agent sessions
├── README.md                     # setup, training, running, troubleshooting
├── pyproject.toml                # deps, ruff, mypy, pytest config
├── docker-compose.yml            # Ollama service only
├── .env.example                  # OLLAMA_HOST, OLLAMA_MODEL, ADC_DATA_DIR
├── .gitignore
├── scripts/
│   ├── check.sh                  # ruff check + ruff format --check + mypy + pytest
│   └── fetch_dataset.py          # download/extract or link the dataset
├── src/aircraft_damage/
│   ├── __init__.py
│   ├── tf_env.py                 # TF env vars; MUST be imported before tensorflow
│   ├── config.py                 # stdlib-only Config dataclass + load_config()
│   ├── data.py                   # ImageDataGenerator builders (notebook 1.1–1.2)
│   ├── model.py                  # VGG16 + dense head, compiled (notebook 1.3)
│   ├── train.py                  # training CLI, writes artifacts (notebook 1.4–1.6)
│   ├── classifier.py             # inference wrapper over the saved .keras model
│   ├── captioning.py             # BlipCaptionSummaryLayer (notebook Part 2) + BlipDescriber
│   ├── llm.py                    # Ollama HTTP client, thinking-tag stripping
│   ├── report.py                 # EvidencePacket, prompts, generate_report
│   ├── pipeline.py               # run_inspection: classify -> describe -> report
│   └── app/
│       ├── __init__.py
│       ├── styles.py             # CUSTOM_CSS constant
│       └── streamlit_app.py      # UI only, no domain logic
└── tests/
    ├── conftest.py               # synthetic-dataset and fake-collaborator fixtures
    ├── test_config.py
    ├── test_data.py
    ├── test_model.py
    ├── test_train.py
    ├── test_classifier.py
    ├── test_captioning.py
    ├── test_llm.py
    ├── test_report.py
    └── test_pipeline.py
```

**Why this split:** `config.py` imports nothing heavy, so Streamlit can read settings
without paying TensorFlow's import cost. `classifier.py`, `report.py`, `llm.py`, and
`pipeline.py` are all testable with fakes and zero ML dependencies at test time.
`captioning.py` and `model.py` are the only modules that must touch TF/torch.
`streamlit_app.py` holds no logic, so the untestable layer stays trivially thin.

---

## Task 1: Project scaffold, conventions, and configuration

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `CLAUDE.md`, `scripts/check.sh`
- Create: `src/aircraft_damage/__init__.py`, `src/aircraft_damage/tf_env.py`, `src/aircraft_damage/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Config` (frozen dataclass with fields `data_dir: Path`, `artifacts_dir: Path`, `img_rows: int`, `img_cols: int`, `batch_size: int`, `n_epochs: int`, `seed: int`, `learning_rate: float`, `blip_model_id: str`, `ollama_host: str`, `ollama_model: str`, `ollama_timeout_s: int`; properties `train_dir`, `valid_dir`, `test_dir`, `target_size -> tuple[int, int]`, `input_shape -> tuple[int, int, int]`, `model_path`, `metrics_path`, `accuracy_curve_path`, `loss_curve_path`) and `load_config() -> Config`. Also `PROJECT_ROOT: Path`.

- [ ] **Step 1: Create the repo and the Python 3.12 venv**

```bash
mkdir -p /home/rensilver/workspaces/aircraft-damage-classification
cd /home/rensilver/workspaces/aircraft-damage-classification
git init -b main
git commit --allow-empty -m "chore: initial commit"
git checkout -b task-01-scaffold
uv venv --python 3.12
```

Expected: `.venv` created reporting `Using CPython 3.12.x`, and `git branch --show-current` prints `task-01-scaffold`. If uv reports it cannot find 3.12, run `uv python install 3.12` first.

The empty initial commit exists so `main` has a root for later `--no-ff` merges. The
spec and this plan already live in `docs/superpowers/`; they get committed with this
task.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "aircraft-damage-classification"
version = "0.1.0"
description = "Classify aircraft damage with VGG16, describe with BLIP, report with a local LLM."
requires-python = "==3.12.*"
dependencies = [
    "tensorflow-cpu==2.17.1",
    "numpy>=1.26,<2",
    "pillow==11.1.0",
    "matplotlib==3.9.2",
    "transformers==4.44.2",
    "torch==2.4.1+cpu",
    "httpx==0.27.2",
    "streamlit==1.39.0",
]

[dependency-groups]
dev = ["pytest==8.3.3", "ruff==0.6.9", "mypy==1.11.2"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/aircraft_damage"]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }

[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "SIM", "PTH", "RET", "ARG", "TID", "ANN", "D"]
ignore = ["D203", "D213", "ANN401"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["D100", "D103", "D101", "D102", "ARG001", "ARG002"]
"scripts/*" = ["D100", "D103"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.mypy]
python_version = "3.12"
files = ["src", "tests"]
disallow_untyped_defs = true
disallow_incomplete_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = false

[[tool.mypy.overrides]]
module = ["tensorflow.*", "keras.*", "transformers.*", "torch.*", "matplotlib.*", "streamlit.*", "PIL.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers"
markers = ["slow: needs model downloads, network, or real training"]
```

- [ ] **Step 3: Install dependencies**

```bash
uv sync
```

Expected: resolves and installs, with `torch` coming from the `pytorch-cpu` index. This downloads ~1 GB; allow several minutes. If `tensorflow-cpu` fails to resolve, confirm the venv is 3.12 with `uv run python -V`.

- [ ] **Step 4: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
data/
artifacts/
.env
uv.lock
```

- [ ] **Step 5: Write `.env.example`**

```bash
# Copy to .env and edit as needed.
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
ADC_DATA_DIR=data/aircraft_damage_dataset_v1
ADC_ARTIFACTS_DIR=artifacts
```

- [ ] **Step 6: Write `CLAUDE.md`**

Copy the fifteen numbered rules from **Global Constraints → Python coding conventions** above into `CLAUDE.md` verbatim, under an `# Aircraft Damage Classification — Conventions` heading, followed by these two sections:

```markdown
## Architecture invariants

- `qwen3:4b` is text-only. It never receives image pixels. Any code or copy
  implying the LLM saw the image is a bug.
- `config.py` imports stdlib only. Do not add TensorFlow, torch, or Streamlit
  imports to it — Streamlit reads config before deciding whether to load models.
- `src/aircraft_damage/app/streamlit_app.py` contains no domain logic. New
  behaviour goes in `pipeline.py` (or lower) with a test, and the app calls it.
- Import `aircraft_damage.tf_env` before any `tensorflow` / `keras` import.

## Commands

- Full check: `./scripts/check.sh`
- Tests only: `uv run pytest -m "not slow"`
- Train: `uv run python -m aircraft_damage.train`
- App: `uv run streamlit run src/aircraft_damage/app/streamlit_app.py`
- Ollama: `docker compose up -d` then `docker compose exec ollama ollama pull qwen3:4b`
```

- [ ] **Step 7: Write `scripts/check.sh` and make it executable**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "==> ruff check"
uv run ruff check src tests scripts

echo "==> ruff format --check"
uv run ruff format --check src tests scripts

echo "==> mypy"
uv run mypy

echo "==> pytest"
uv run pytest -m "not slow"

echo "All checks passed."
```

Then: `chmod +x scripts/check.sh`

- [ ] **Step 8: Write the failing test for config**

Create `tests/test_config.py`:

```python
from __future__ import annotations

from pathlib import Path

from aircraft_damage.config import Config, load_config


def _config(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")


def test_defaults_match_the_notebook(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert config.seed == 42
    assert config.batch_size == 32
    assert config.n_epochs == 5
    assert config.learning_rate == 0.0001
    assert config.target_size == (224, 224)
    assert config.input_shape == (224, 224, 3)


def test_split_directories_hang_off_the_data_dir(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert config.train_dir == tmp_path / "data" / "train"
    assert config.valid_dir == tmp_path / "data" / "valid"
    assert config.test_dir == tmp_path / "data" / "test"


def test_artifact_paths_hang_off_the_artifacts_dir(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert config.model_path == tmp_path / "artifacts" / "vgg16_damage_classifier.keras"
    assert config.metrics_path == tmp_path / "artifacts" / "metrics.json"
    assert config.accuracy_curve_path == tmp_path / "artifacts" / "accuracy_curve.png"
    assert config.loss_curve_path == tmp_path / "artifacts" / "loss_curve.png"


def test_config_is_immutable(tmp_path: Path) -> None:
    config = _config(tmp_path)

    try:
        config.seed = 7  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("Config should be frozen")


def test_load_config_reads_ollama_settings_from_the_environment(
    monkeypatch: object,
) -> None:
    import os

    os.environ["OLLAMA_HOST"] = "http://ollama.test:11434"
    os.environ["OLLAMA_MODEL"] = "qwen3:8b"
    try:
        config = load_config()
        assert config.ollama_host == "http://ollama.test:11434"
        assert config.ollama_model == "qwen3:8b"
    finally:
        del os.environ["OLLAMA_HOST"]
        del os.environ["OLLAMA_MODEL"]


def test_load_config_falls_back_to_local_ollama() -> None:
    config = load_config()

    assert config.ollama_host == "http://localhost:11434"
    assert config.ollama_model == "qwen3:4b"
```

- [ ] **Step 9: Run the test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aircraft_damage.config'`

- [ ] **Step 10: Write `src/aircraft_damage/__init__.py`**

```python
"""Aircraft damage classification, description, and reporting."""

from __future__ import annotations

__version__ = "0.1.0"
```

- [ ] **Step 11: Write `src/aircraft_damage/tf_env.py`**

```python
"""TensorFlow environment flags.

Import this module *before* importing ``tensorflow`` or ``keras``; the variables
below are only read at TensorFlow import time. Mirrors the notebook's setup cell.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
```

- [ ] **Step 12: Write `src/aircraft_damage/config.py`**

```python
"""Project configuration.

This module imports the standard library only. Streamlit reads configuration
before deciding whether to pay TensorFlow's import cost, so keep it that way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "aircraft_damage_dataset_v1"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
MODEL_FILENAME = "vgg16_damage_classifier.keras"
METRICS_FILENAME = "metrics.json"
ACCURACY_CURVE_FILENAME = "accuracy_curve.png"
LOSS_CURVE_FILENAME = "loss_curve.png"


@dataclass(frozen=True)
class Config:
    """Immutable settings for training, inference, and reporting.

    Defaults are copied verbatim from the source notebook so that a training run
    here is comparable to the graded notebook run.
    """

    data_dir: Path
    artifacts_dir: Path
    img_rows: int = 224
    img_cols: int = 224
    batch_size: int = 32
    n_epochs: int = 5
    seed: int = 42
    learning_rate: float = 0.0001
    blip_model_id: str = "Salesforce/blip-image-captioning-base"
    ollama_host: str = DEFAULT_OLLAMA_HOST
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_timeout_s: int = 180

    @property
    def train_dir(self) -> Path:
        """Directory holding the training split."""
        return self.data_dir / "train"

    @property
    def valid_dir(self) -> Path:
        """Directory holding the validation split."""
        return self.data_dir / "valid"

    @property
    def test_dir(self) -> Path:
        """Directory holding the test split."""
        return self.data_dir / "test"

    @property
    def target_size(self) -> tuple[int, int]:
        """Image size, as Keras generators expect it."""
        return (self.img_rows, self.img_cols)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Model input shape, including the channel dimension."""
        return (self.img_rows, self.img_cols, 3)

    @property
    def model_path(self) -> Path:
        """Path of the saved Keras classifier."""
        return self.artifacts_dir / MODEL_FILENAME

    @property
    def metrics_path(self) -> Path:
        """Path of the training metrics JSON."""
        return self.artifacts_dir / METRICS_FILENAME

    @property
    def accuracy_curve_path(self) -> Path:
        """Path of the accuracy curve image."""
        return self.artifacts_dir / ACCURACY_CURVE_FILENAME

    @property
    def loss_curve_path(self) -> Path:
        """Path of the loss curve image."""
        return self.artifacts_dir / LOSS_CURVE_FILENAME


def load_config() -> Config:
    """Build a :class:`Config` from environment variables, falling back to defaults.

    Returns:
        The configuration for this process.
    """
    return Config(
        data_dir=Path(os.environ.get("ADC_DATA_DIR", DEFAULT_DATA_DIR)),
        artifacts_dir=Path(os.environ.get("ADC_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)),
        ollama_host=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        ollama_model=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
    )
```

- [ ] **Step 13: Run the test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 14: Run the full check script**

Run: `./scripts/check.sh`
Expected: `All checks passed.` If `ruff format --check` fails, run `uv run ruff format src tests scripts` and re-run.

- [ ] **Step 15: Commit and merge the task branch**

```bash
git add -A
git commit -m "chore: scaffold project with config, tooling, and conventions"
git checkout main
git merge --no-ff task-01-scaffold -m "merge: task 01 project scaffold and configuration"
```

Expected: `git log --oneline --graph -5` on `main` shows the merge commit above the
task commit.

---

## Task 2: Dataset acquisition and Keras data generators

**Files:**
- Create: `scripts/fetch_dataset.py`, `src/aircraft_damage/data.py`
- Test: `tests/conftest.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `Config` from Task 1 (uses `train_dir`, `valid_dir`, `test_dir`, `target_size`, `batch_size`, `seed`).
- Produces: `build_generators(train_dir: Path, valid_dir: Path, test_dir: Path, *, target_size: tuple[int, int], batch_size: int, seed: int) -> tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]` returning `(train, valid, test)` in that order, and `class_names_from_indices(class_indices: dict[str, int]) -> dict[int, str]`.

- [ ] **Step 1: Create the task branch**

```bash
git checkout main
git checkout -b task-02-data
```

- [ ] **Step 2: Write `tests/conftest.py` with a synthetic-dataset fixture**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

CLASS_NAMES = ("crack", "dent")
SPLITS = ("train", "valid", "test")
IMAGES_PER_CLASS = 3


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Build a tiny on-disk dataset with the real folder layout.

    Returns:
        Root directory containing ``train/``, ``valid/`` and ``test/`` splits,
        each with ``crack/`` and ``dent/`` subfolders of small RGB JPEGs.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "dataset"
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            folder = root / split / class_name
            folder.mkdir(parents=True)
            for index in range(IMAGES_PER_CLASS):
                pixels = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{class_name}_{index}.jpg")
    return root


@pytest.fixture
def sample_image() -> Image.Image:
    """A single small RGB image, for inference-side tests."""
    rng = np.random.default_rng(1)
    pixels = rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)
    return Image.fromarray(pixels)
```

- [ ] **Step 3: Write the failing test for the generators**

Create `tests/test_data.py`:

```python
from __future__ import annotations

from pathlib import Path

from aircraft_damage.data import build_generators, class_names_from_indices


def test_generators_discover_both_classes_in_alphabetical_order(
    synthetic_dataset: Path,
) -> None:
    train, valid, test = build_generators(
        synthetic_dataset / "train",
        synthetic_dataset / "valid",
        synthetic_dataset / "test",
        target_size=(224, 224),
        batch_size=4,
        seed=42,
    )

    assert train.class_indices == {"crack": 0, "dent": 1}
    assert valid.class_indices == {"crack": 0, "dent": 1}
    assert test.class_indices == {"crack": 0, "dent": 1}


def test_generators_find_every_image(synthetic_dataset: Path) -> None:
    train, valid, test = build_generators(
        synthetic_dataset / "train",
        synthetic_dataset / "valid",
        synthetic_dataset / "test",
        target_size=(224, 224),
        batch_size=4,
        seed=42,
    )

    assert (train.samples, valid.samples, test.samples) == (6, 6, 6)


def test_batches_are_resized_and_rescaled(synthetic_dataset: Path) -> None:
    train, _, _ = build_generators(
        synthetic_dataset / "train",
        synthetic_dataset / "valid",
        synthetic_dataset / "test",
        target_size=(224, 224),
        batch_size=4,
        seed=42,
    )

    images, labels = next(train)

    assert images.shape[1:] == (224, 224, 3)
    assert images.max() <= 1.0
    assert set(labels.tolist()) <= {0.0, 1.0}


def test_only_the_training_generator_shuffles(synthetic_dataset: Path) -> None:
    train, valid, test = build_generators(
        synthetic_dataset / "train",
        synthetic_dataset / "valid",
        synthetic_dataset / "test",
        target_size=(224, 224),
        batch_size=4,
        seed=42,
    )

    assert train.shuffle is True
    assert valid.shuffle is False
    assert test.shuffle is False


def test_class_names_from_indices_inverts_the_mapping() -> None:
    assert class_names_from_indices({"crack": 0, "dent": 1}) == {0: "crack", 1: "dent"}
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aircraft_damage.data'`

- [ ] **Step 5: Write `src/aircraft_damage/data.py`**

```python
"""Keras image generators for the aircraft damage dataset.

Mirrors sections 1.1 and 1.2 of the source notebook.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from aircraft_damage import tf_env  # noqa: F401  # must precede the tensorflow import

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # noqa: E402

if TYPE_CHECKING:
    from tensorflow.keras.preprocessing.image import DirectoryIterator

logger = logging.getLogger(__name__)

RESCALE_FACTOR = 1.0 / 255
CLASS_MODE = "binary"


def build_generators(
    train_dir: Path,
    valid_dir: Path,
    test_dir: Path,
    *,
    target_size: tuple[int, int],
    batch_size: int,
    seed: int,
) -> tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
    """Create the train, validation, and test generators.

    Only the training generator shuffles; validation and test keep file order so
    that predictions line up with ``generator.classes``.

    Args:
        train_dir: Directory containing the training split.
        valid_dir: Directory containing the validation split.
        test_dir: Directory containing the test split.
        target_size: Height and width every image is resized to.
        batch_size: Images per batch.
        seed: Shuffle seed, for reproducibility.

    Returns:
        The train, validation, and test generators, in that order.
    """
    datagen = ImageDataGenerator(rescale=RESCALE_FACTOR)

    def flow(directory: Path, *, shuffle: bool) -> DirectoryIterator:
        return datagen.flow_from_directory(
            directory=str(directory),
            class_mode=CLASS_MODE,
            seed=seed,
            batch_size=batch_size,
            shuffle=shuffle,
            target_size=target_size,
        )

    train_generator = flow(train_dir, shuffle=True)
    valid_generator = flow(valid_dir, shuffle=False)
    test_generator = flow(test_dir, shuffle=False)

    logger.info(
        "Loaded %d train / %d valid / %d test images across classes %s",
        train_generator.samples,
        valid_generator.samples,
        test_generator.samples,
        train_generator.class_indices,
    )
    return train_generator, valid_generator, test_generator


def class_names_from_indices(class_indices: dict[str, int]) -> dict[int, str]:
    """Invert a Keras ``class_indices`` mapping.

    Args:
        class_indices: Mapping of class name to index, as Keras produces it.

    Returns:
        Mapping of index to class name.
    """
    return {index: name for name, index in class_indices.items()}
```

Note: the notebook creates three separate `ImageDataGenerator` instances with identical
settings. One shared instance is equivalent — `ImageDataGenerator` holds no per-flow
state — and keeps the module honest about there being no augmentation.

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_data.py -v`
Expected: PASS, 5 tests. First run is slow (TensorFlow import ~10 s).

- [ ] **Step 7: Write `scripts/fetch_dataset.py`**

```python
"""Download and extract the aircraft damage dataset.

Run with ``uv run python scripts/fetch_dataset.py``. If you already have the
dataset extracted elsewhere, pass ``--link-from`` to symlink it instead.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

DATASET_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud"
    "/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "aircraft_damage_dataset_v1"
EXPECTED_SPLITS = ("train", "valid", "test")
EXPECTED_CLASSES = ("crack", "dent")


def verify(dataset_dir: Path) -> None:
    """Check the extracted layout and print per-split counts.

    Args:
        dataset_dir: Root of the extracted dataset.

    Raises:
        SystemExit: If an expected split or class folder is missing.
    """
    for split in EXPECTED_SPLITS:
        for class_name in EXPECTED_CLASSES:
            folder = dataset_dir / split / class_name
            if not folder.is_dir():
                sys.exit(f"Missing expected folder: {folder}")
            count = len(list(folder.glob("*.jpg")))
            print(f"  {split}/{class_name}: {count} images")


def link_from(source: Path) -> None:
    """Symlink an existing extracted dataset into ``data/``.

    Args:
        source: Path to an existing ``aircraft_damage_dataset_v1`` directory.

    Raises:
        SystemExit: If the source does not exist.
    """
    if not source.is_dir():
        sys.exit(f"Source directory does not exist: {source}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_DIR.exists() or DATASET_DIR.is_symlink():
        sys.exit(f"{DATASET_DIR} already exists; remove it first.")
    DATASET_DIR.symlink_to(source.resolve(), target_is_directory=True)
    print(f"Linked {DATASET_DIR} -> {source.resolve()}")


def download() -> None:
    """Download the tarball and extract it into ``data/``."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = DATA_DIR / "aircraft_damage_dataset_v1.tar"

    print(f"Downloading {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, tar_path)  # noqa: S310

    if DATASET_DIR.exists():
        print(f"Removing existing {DATASET_DIR}")
        shutil.rmtree(DATASET_DIR)

    print(f"Extracting into {DATA_DIR}")
    with tarfile.open(tar_path, "r") as archive:
        archive.extractall(DATA_DIR, filter="data")
    tar_path.unlink()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--link-from",
        type=Path,
        default=None,
        help="Symlink an already-extracted dataset instead of downloading.",
    )
    args = parser.parse_args()

    if args.link_from is not None:
        link_from(args.link_from)
    elif DATASET_DIR.exists():
        print(f"{DATASET_DIR} already exists; skipping download.")
    else:
        download()

    print("Dataset layout:")
    verify(DATASET_DIR)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Get the dataset in place**

The dataset is already extracted on this machine, so linking is faster than downloading:

```bash
uv run python scripts/fetch_dataset.py --link-from /home/rensilver/workspaces/ibm-ai-engineering/02_deep_learning_neural_networks_keras/data/aircraft_damage_dataset_v1
```

Expected output ends with `train/crack: 150`, `train/dent: 150`, `valid/crack: 48`,
`valid/dent: 48`, `test/crack: 25`, `test/dent: 25`.

- [ ] **Step 9: Run the full check script**

Run: `./scripts/check.sh`
Expected: `All checks passed.`

- [ ] **Step 10: Commit and merge the task branch**

```bash
git add -A
git commit -m "feat: add dataset fetch script and Keras data generators"
git checkout main
git merge --no-ff task-02-data -m "merge: task 02 dataset and data generators"
```

---

## Task 3: VGG16 model definition

**Files:**
- Create: `src/aircraft_damage/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (takes `input_shape` and `learning_rate` as arguments).
- Produces: `build_feature_extractor(input_shape: tuple[int, int, int], *, weights: str | None = "imagenet") -> Model` and `build_model(input_shape: tuple[int, int, int], *, learning_rate: float, weights: str | None = "imagenet") -> Sequential`. The `weights` keyword exists so tests can pass `None` and skip the 58 MB ImageNet download.

- [ ] **Step 1: Create the task branch**

```bash
git checkout main
git checkout -b task-03-model
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_model.py`:

```python
from __future__ import annotations

import numpy as np

from aircraft_damage.model import build_feature_extractor, build_model

INPUT_SHAPE = (224, 224, 3)


def test_feature_extractor_is_entirely_frozen() -> None:
    extractor = build_feature_extractor(INPUT_SHAPE, weights=None)

    assert all(not layer.trainable for layer in extractor.layers)


def test_feature_extractor_flattens_to_the_vgg16_feature_vector() -> None:
    extractor = build_feature_extractor(INPUT_SHAPE, weights=None)

    assert extractor.output_shape == (None, 7 * 7 * 512)


def test_model_has_the_notebook_topology() -> None:
    model = build_model(INPUT_SHAPE, learning_rate=0.0001, weights=None)

    layer_types = [type(layer).__name__ for layer in model.layers]
    assert layer_types == [
        "Functional",
        "Dense",
        "Dropout",
        "Dense",
        "Dropout",
        "Dense",
    ]


def test_model_emits_a_single_sigmoid_probability() -> None:
    model = build_model(INPUT_SHAPE, learning_rate=0.0001, weights=None)

    assert model.output_shape == (None, 1)
    assert model.layers[-1].activation.__name__ == "sigmoid"


def test_model_is_compiled_with_the_notebook_optimiser_and_loss() -> None:
    model = build_model(INPUT_SHAPE, learning_rate=0.0001, weights=None)

    assert model.optimizer.name.lower() == "adam"
    assert float(model.optimizer.learning_rate) == 0.0001
    assert model.loss == "binary_crossentropy"


def test_only_the_classifier_head_is_trainable() -> None:
    model = build_model(INPUT_SHAPE, learning_rate=0.0001, weights=None)

    frozen = model.layers[0].count_params()
    assert frozen > 14_000_000
    assert sum(int(np.prod(w.shape)) for w in model.layers[0].trainable_weights) == 0


def test_prediction_is_a_probability_per_image() -> None:
    model = build_model(INPUT_SHAPE, learning_rate=0.0001, weights=None)

    predictions = model.predict(np.zeros((2, *INPUT_SHAPE), dtype=np.float32), verbose=0)

    assert predictions.shape == (2, 1)
    assert np.all((predictions >= 0.0) & (predictions <= 1.0))
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aircraft_damage.model'`

- [ ] **Step 4: Write `src/aircraft_damage/model.py`**

```python
"""VGG16 feature extractor plus a binary classification head.

Mirrors section 1.3 of the source notebook: the ImageNet-pretrained VGG16
convolutional stack is frozen and used purely as a feature extractor, and only
the dense head on top is trained.
"""

from __future__ import annotations

import logging

from aircraft_damage import tf_env  # noqa: F401  # must precede the keras import

from keras.applications import VGG16  # noqa: E402
from keras.layers import Dense, Dropout, Flatten  # noqa: E402
from keras.models import Model, Sequential  # noqa: E402
from keras.optimizers import Adam  # noqa: E402

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
    logger.info("Built classifier with %d trainable parameters", model.count_params())
    return model
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_model.py -v`
Expected: PASS, 7 tests. Takes ~30 s — seven VGG16 graphs get built.

If `test_model_has_the_notebook_topology` fails on the first entry, print the actual
list and adjust the expected string to whatever Keras 3 names a nested functional
model in this version. Do not change the topology to satisfy the test.

- [ ] **Step 6: Run the full check script**

Run: `./scripts/check.sh`
Expected: `All checks passed.`

- [ ] **Step 7: Commit and merge the task branch**

```bash
git add -A
git commit -m "feat: add frozen VGG16 feature extractor and classification head"
git checkout main
git merge --no-ff task-03-model -m "merge: task 03 VGG16 model definition"
```

---

## Task 4: Training CLI and artifacts

**Files:**
- Create: `src/aircraft_damage/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `Config` (Task 1), `build_generators` (Task 2), `build_model` (Task 3).
- Produces: `set_seeds(seed: int) -> None`, `plot_curves(history: dict[str, list[float]], accuracy_path: Path, loss_path: Path) -> None`, `save_metrics(path: Path, *, history: dict[str, list[float]], test_loss: float, test_accuracy: float, class_indices: dict[str, int], epochs: int, seed: int) -> dict[str, Any]`, `run_training(config: Config, *, weights: str | None = "imagenet") -> dict[str, Any]`, and `main() -> None`. The metrics dict written to `metrics.json` has keys `history`, `test_loss`, `test_accuracy`, `class_indices`, `epochs`, `seed` — Task 5 reads `class_indices` and Task 10 reads `test_accuracy`.

**One deliberate fix to the notebook:** the notebook evaluates with
`steps=test_generator.samples // test_generator.batch_size`, which is `50 // 32 == 1`
and therefore scores only 32 of the 50 test images. `run_training` evaluates the full
test set. This is a bug fix, not a fidelity break; it is recorded in the spec.

- [ ] **Step 1: Create the task branch**

```bash
git checkout main
git checkout -b task-04-train
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_train.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aircraft_damage.train import plot_curves, run_training, save_metrics, set_seeds

HISTORY = {
    "loss": [0.7, 0.5, 0.3],
    "val_loss": [0.8, 0.6, 0.5],
    "accuracy": [0.5, 0.7, 0.9],
    "val_accuracy": [0.4, 0.6, 0.8],
}


def test_set_seeds_makes_numpy_reproducible() -> None:
    set_seeds(42)
    first = np.random.rand(5)
    set_seeds(42)
    second = np.random.rand(5)

    assert np.array_equal(first, second)


def test_plot_curves_writes_both_images(tmp_path: Path) -> None:
    accuracy_path = tmp_path / "accuracy_curve.png"
    loss_path = tmp_path / "loss_curve.png"

    plot_curves(HISTORY, accuracy_path, loss_path)

    assert accuracy_path.stat().st_size > 0
    assert loss_path.stat().st_size > 0


def test_save_metrics_writes_every_required_key(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "metrics.json"

    save_metrics(
        path,
        history=HISTORY,
        test_loss=0.31,
        test_accuracy=0.88,
        class_indices={"crack": 0, "dent": 1},
        epochs=5,
        seed=42,
    )

    written = json.loads(path.read_text())
    assert set(written) == {
        "history",
        "test_loss",
        "test_accuracy",
        "class_indices",
        "epochs",
        "seed",
    }
    assert written["class_indices"] == {"crack": 0, "dent": 1}
    assert written["test_accuracy"] == pytest.approx(0.88)


def test_save_metrics_coerces_numpy_floats_to_json_safe_values(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"

    save_metrics(
        path,
        history={"loss": [np.float32(0.5), np.float32(0.25)]},
        test_loss=np.float32(0.31),
        test_accuracy=np.float32(0.88),
        class_indices={"crack": 0, "dent": 1},
        epochs=5,
        seed=42,
    )

    written = json.loads(path.read_text())
    assert written["history"]["loss"] == [0.5, 0.25]
    assert isinstance(written["test_loss"], float)


@pytest.mark.slow
def test_run_training_produces_every_artifact(synthetic_dataset: Path, tmp_path: Path) -> None:
    from dataclasses import replace

    from aircraft_damage.config import Config

    config = replace(
        Config(data_dir=synthetic_dataset, artifacts_dir=tmp_path / "artifacts"),
        n_epochs=1,
        batch_size=2,
    )

    metrics = run_training(config, weights=None)

    assert config.model_path.exists()
    assert config.metrics_path.exists()
    assert config.accuracy_curve_path.exists()
    assert config.loss_curve_path.exists()
    assert metrics["class_indices"] == {"crack": 0, "dent": 1}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_train.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'aircraft_damage.train'`

- [ ] **Step 4: Write `src/aircraft_damage/train.py`**

```python
"""Train the aircraft damage classifier and write artifacts.

Mirrors sections 1.4 to 1.6 of the source notebook. Run with::

    uv run python -m aircraft_damage.train
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

from aircraft_damage import tf_env  # noqa: F401  # must precede the tensorflow import

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from aircraft_damage.config import Config, load_config  # noqa: E402
from aircraft_damage.data import build_generators  # noqa: E402
from aircraft_damage.model import build_model  # noqa: E402

logger = logging.getLogger(__name__)

FIGURE_SIZE = (5, 5)


def set_seeds(seed: int) -> None:
    """Seed every random source the training run touches.

    Args:
        seed: The seed value; 42 in the source notebook.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def plot_curves(
    history: dict[str, list[float]],
    accuracy_path: Path,
    loss_path: Path,
) -> None:
    """Write accuracy and loss curve images.

    Args:
        history: Keras ``History.history`` mapping metric name to per-epoch values.
        accuracy_path: Destination for the accuracy curve.
        loss_path: Destination for the loss curve.
    """
    accuracy_path.parent.mkdir(parents=True, exist_ok=True)

    for title, ylabel, keys, path in (
        ("Accuracy Curve", "Accuracy", ("accuracy", "val_accuracy"), accuracy_path),
        ("Loss Curve", "Loss", ("loss", "val_loss"), loss_path),
    ):
        figure = plt.figure(figsize=FIGURE_SIZE)
        for key in keys:
            if key in history:
                plt.plot(history[key], label=key.replace("_", " ").title())
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend()
        figure.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(figure)

    logger.info("Wrote curves to %s and %s", accuracy_path, loss_path)


def save_metrics(
    path: Path,
    *,
    history: dict[str, list[float]],
    test_loss: float,
    test_accuracy: float,
    class_indices: dict[str, int],
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    """Write the training metrics JSON.

    Keras history values are numpy floats, which ``json`` cannot serialise, so
    everything numeric is coerced to a Python float first.

    Args:
        path: Destination file; parent directories are created.
        history: Keras ``History.history``.
        test_loss: Loss on the held-out test split.
        test_accuracy: Accuracy on the held-out test split.
        class_indices: The generator's class-name-to-index mapping.
        epochs: Number of epochs trained.
        seed: Seed used for the run.

    Returns:
        The dictionary that was written.
    """
    metrics: dict[str, Any] = {
        "history": {key: [float(value) for value in values] for key, values in history.items()},
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "class_indices": dict(class_indices),
        "epochs": int(epochs),
        "seed": int(seed),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote metrics to %s", path)
    return metrics


def run_training(config: Config, *, weights: str | None = "imagenet") -> dict[str, Any]:
    """Run the full training, evaluation, and artifact-writing cycle.

    Args:
        config: Paths and hyperparameters for the run.
        weights: Passed to the feature extractor; ``None`` in fast tests.

    Returns:
        The metrics dictionary, as written to ``config.metrics_path``.
    """
    set_seeds(config.seed)

    train_generator, valid_generator, test_generator = build_generators(
        config.train_dir,
        config.valid_dir,
        config.test_dir,
        target_size=config.target_size,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    model = build_model(
        config.input_shape,
        learning_rate=config.learning_rate,
        weights=weights,
    )

    history = model.fit(
        train_generator,
        epochs=config.n_epochs,
        validation_data=valid_generator,
    )

    # The notebook passes steps=samples//batch_size here, which silently drops the
    # final partial batch. Evaluating the whole split is the correct behaviour.
    test_loss, test_accuracy = model.evaluate(test_generator)
    logger.info("Test loss %.4f, test accuracy %.4f", test_loss, test_accuracy)

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(config.model_path)
    logger.info("Saved model to %s", config.model_path)

    plot_curves(history.history, config.accuracy_curve_path, config.loss_curve_path)

    return save_metrics(
        config.metrics_path,
        history=history.history,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        class_indices=train_generator.class_indices,
        epochs=config.n_epochs,
        seed=config.seed,
    )


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train the aircraft damage classifier.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch count.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override the dataset root.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Override the artifacts output directory.",
    )
    args = parser.parse_args()

    config = load_config()
    if args.epochs is not None:
        config = replace(config, n_epochs=args.epochs)
    if args.data_dir is not None:
        config = replace(config, data_dir=args.data_dir)
    if args.artifacts_dir is not None:
        config = replace(config, artifacts_dir=args.artifacts_dir)

    metrics = run_training(config)

    print(f"Test loss:     {metrics['test_loss']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Artifacts in:  {config.artifacts_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the fast tests to verify they pass**

Run: `uv run pytest tests/test_train.py -v -m "not slow"`
Expected: PASS, 4 tests.

- [ ] **Step 6: Run the slow integration test**

Run: `uv run pytest tests/test_train.py -v -m slow`
Expected: PASS, 1 test. Takes ~1–2 min on CPU.

- [ ] **Step 7: Train the real model**

```bash
uv run python -m aircraft_damage.train
```

Expected: 5 epochs over 300 images (~10 batches each), then a test evaluation.
On 8 CPU cores this takes roughly 10–20 minutes. Confirm afterwards:

```bash
ls -lh artifacts/
```

Expected: `vgg16_damage_classifier.keras`, `metrics.json`, `accuracy_curve.png`,
`loss_curve.png`. Open `metrics.json` and record `test_accuracy` — Task 10 surfaces it
in the app and Task 8 feeds it to the LLM.

If the process is killed by the OOM reaper, re-run with `--epochs 5` after closing
other applications; 300 images at batch 32 needs roughly 2.5 GB resident.

- [ ] **Step 8: Run the full check script**

Run: `./scripts/check.sh`
Expected: `All checks passed.` (Artifacts are gitignored and will not be committed.)

- [ ] **Step 9: Commit and merge the task branch**

```bash
git add -A
git commit -m "feat: add training CLI writing model, metrics, and curve artifacts"
git checkout main
git merge --no-ff task-04-train -m "merge: task 04 training pipeline"
```

---

## Task 5: Single-image classifier for inference

**Files:**
- Create: `src/aircraft_damage/classifier.py`
- Test: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `class_names_from_indices` (Task 2); the `metrics.json` `class_indices` key and the `.keras` file (Task 4).
- Produces: `ClassificationResult` (frozen dataclass: `label: str`, `confidence: float`, `probabilities: dict[str, float]`), `ModelNotTrainedError(FileNotFoundError)`, and `DamageClassifier` with `__init__(model, class_names: dict[int, str], target_size: tuple[int, int] = (224, 224))`, `classmethod load(model_path: Path, metrics_path: Path) -> DamageClassifier`, `preprocess(image: Image.Image) -> np.ndarray`, and `predict(image: Image.Image) -> ClassificationResult`. Task 9 calls `predict`; Task 10 calls `load`.

This module must stay importable without TensorFlow — the Keras import lives inside
`load()`. That is what lets these tests run in under a second with a stub model.

- [ ] **Step 1: Create the task branch**

```bash
git checkout main
git checkout -b task-05-classifier
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_classifier.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from aircraft_damage.classifier import (
    ClassificationResult,
    DamageClassifier,
    ModelNotTrainedError,
)

CLASS_NAMES = {0: "crack", 1: "dent"}


class StubModel:
    """A stand-in for a Keras model that always returns a fixed probability."""

    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.last_batch: Any = None

    def predict(self, batch: Any, verbose: int = 0) -> Any:
        self.last_batch = batch
        return np.array([[self.probability]], dtype=np.float32)


def test_preprocess_produces_one_normalised_224_square(sample_image: Image.Image) -> None:
    classifier = DamageClassifier(StubModel(0.5), CLASS_NAMES)

    batch = classifier.preprocess(sample_image)

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert batch.max() <= 1.0
    assert batch.min() >= 0.0


def test_preprocess_converts_greyscale_to_rgb() -> None:
    classifier = DamageClassifier(StubModel(0.5), CLASS_NAMES)
    grey = Image.new("L", (30, 40), color=128)

    batch = classifier.preprocess(grey)

    assert batch.shape == (1, 224, 224, 3)


def test_high_probability_selects_the_class_at_index_one(sample_image: Image.Image) -> None:
    classifier = DamageClassifier(StubModel(0.87), CLASS_NAMES)

    result = classifier.predict(sample_image)

    assert result.label == "dent"
    assert result.confidence == pytest.approx(0.87)


def test_low_probability_selects_the_class_at_index_zero(sample_image: Image.Image) -> None:
    classifier = DamageClassifier(StubModel(0.12), CLASS_NAMES)

    result = classifier.predict(sample_image)

    assert result.label == "crack"
    assert result.confidence == pytest.approx(0.88)


def test_probabilities_cover_both_classes_and_sum_to_one(sample_image: Image.Image) -> None:
    classifier = DamageClassifier(StubModel(0.3), CLASS_NAMES)

    result = classifier.predict(sample_image)

    assert set(result.probabilities) == {"crack", "dent"}
    assert result.probabilities["dent"] == pytest.approx(0.3)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_result_is_immutable(sample_image: Image.Image) -> None:
    result = ClassificationResult(label="dent", confidence=0.9, probabilities={"dent": 0.9})

    with pytest.raises(AttributeError):
        result.label = "crack"  # type: ignore[misc]


def test_load_raises_a_clear_error_when_the_model_is_missing(tmp_path: Path) -> None:
    with pytest.raises(ModelNotTrainedError, match="train"):
        DamageClassifier.load(tmp_path / "missing.keras", tmp_path / "metrics.json")
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aircraft_damage.classifier'`

- [ ] **Step 4: Write `src/aircraft_damage/classifier.py`**

```python
"""Single-image inference over the trained damage classifier.

Deliberately free of TensorFlow at import time: the Keras import lives inside
:meth:`DamageClassifier.load`, so Streamlit and the unit tests can import this
module without paying for TensorFlow.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from aircraft_damage.data import class_names_from_indices

logger = logging.getLogger(__name__)

DEFAULT_TARGET_SIZE = (224, 224)
DECISION_THRESHOLD = 0.5


class ModelNotTrainedError(FileNotFoundError):
    """Raised when the trained classifier artifact cannot be found."""


class PredictsProbabilities(Protocol):
    """The slice of the Keras model interface this module relies on."""

    def predict(self, batch: Any, verbose: int = ...) -> Any:
        """Return a ``(batch, 1)`` array of probabilities."""


@dataclass(frozen=True)
class ClassificationResult:
    """The classifier's verdict for one image.

    Attributes:
        label: The predicted class name.
        confidence: Probability assigned to ``label``, always in ``[0.5, 1.0]``.
        probabilities: Probability for every class name.
    """

    label: str
    confidence: float
    probabilities: dict[str, float]


class DamageClassifier:
    """Wraps a trained Keras model with the notebook's preprocessing."""

    def __init__(
        self,
        model: PredictsProbabilities,
        class_names: dict[int, str],
        target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    ) -> None:
        """Initialise the classifier.

        Args:
            model: Anything exposing Keras' ``predict``.
            class_names: Index-to-name mapping, from ``class_names_from_indices``.
            target_size: Height and width the model expects.
        """
        self._model = model
        self._class_names = class_names
        self._target_size = target_size

    @classmethod
    def load(cls, model_path: Path, metrics_path: Path) -> DamageClassifier:
        """Load a trained classifier and its class mapping from disk.

        Args:
            model_path: Path to the saved ``.keras`` model.
            metrics_path: Path to ``metrics.json``, read for ``class_indices``.

        Returns:
            A ready-to-use classifier.

        Raises:
            ModelNotTrainedError: If either artifact is missing.
        """
        if not model_path.exists():
            raise ModelNotTrainedError(
                f"No trained model at {model_path}. "
                "Run 'uv run python -m aircraft_damage.train' first."
            )
        if not metrics_path.exists():
            raise ModelNotTrainedError(
                f"No metrics at {metrics_path}. "
                "Run 'uv run python -m aircraft_damage.train' first."
            )

        from aircraft_damage import tf_env  # noqa: F401, PLC0415

        import keras  # noqa: PLC0415

        model = keras.saving.load_model(model_path)
        class_indices: dict[str, int] = json.loads(metrics_path.read_text())["class_indices"]
        logger.info("Loaded classifier from %s with classes %s", model_path, class_indices)
        return cls(model, class_names_from_indices(class_indices))

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize, convert, and rescale an image into a single-item batch.

        Args:
            image: Any PIL image, any mode, any size.

        Returns:
            A ``(1, height, width, 3)`` float32 array with values in ``[0, 1]``.
        """
        height, width = self._target_size
        resized = image.convert("RGB").resize((width, height))  # PIL takes (w, h)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        return array[np.newaxis, ...]

    def predict(self, image: Image.Image) -> ClassificationResult:
        """Classify one image.

        Args:
            image: The image to classify.

        Returns:
            The label, its confidence, and the full probability distribution.
        """
        batch = self.preprocess(image)
        probability = float(self._model.predict(batch, verbose=0)[0][0])

        negative, positive = self._class_names[0], self._class_names[1]
        is_positive = probability > DECISION_THRESHOLD

        return ClassificationResult(
            label=positive if is_positive else negative,
            confidence=probability if is_positive else 1.0 - probability,
            probabilities={negative: 1.0 - probability, positive: probability},
        )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: PASS, 7 tests, in well under a second.

- [ ] **Step 6: Sanity-check against the real trained model**

```bash
uv run python -c "
from pathlib import Path
from PIL import Image
from aircraft_damage.classifier import DamageClassifier
from aircraft_damage.config import load_config

config = load_config()
classifier = DamageClassifier.load(config.model_path, config.metrics_path)
image_path = next((config.test_dir / 'dent').glob('*.jpg'))
print(image_path.name, classifier.predict(Image.open(image_path)))
"
```

Expected: a `ClassificationResult` printed with a label of `dent` or `crack` and a
confidence between 0.5 and 1.0. A wrong label on a single image is not a failure.

- [ ] **Step 7: Run the full check script**

Run: `./scripts/check.sh`
Expected: `All checks passed.`

- [ ] **Step 8: Commit and merge the task branch**

```bash
git add -A
git commit -m "feat: add single-image damage classifier for inference"
git checkout main
git merge --no-ff task-05-classifier -m "merge: task 05 inference classifier"
```

---

## Task 6: BLIP captioning and description

**Files:**
- Create: `src/aircraft_damage/captioning.py`
- Test: `tests/test_captioning.py`

**Interfaces:**
- Consumes: `Config.blip_model_id` (Task 1).
- Produces: `ImageDescription` (frozen dataclass: `caption: str`, `summary: str`), `BlipDescriber` with `__init__(processor, model)`, `classmethod load(model_id: str) -> BlipDescriber`, and `describe(image: Image.Image) -> ImageDescription`; plus the notebook's `BlipCaptionSummaryLayer` and `generate_text(image_path, task, processor, model)` for parity with the graded exercise. Task 9 calls `describe`; Task 10 calls `load`.

The notebook's custom Keras layer is preserved verbatim in spirit — it still takes
path tensors through `tf.py_function` — because it is the graded Task 8 artifact. The
app does not use it; `BlipDescriber` works on in-memory PIL images, which is what an
upload gives you.

- [ ] **Step 1: Create the task branch**

```bash
git checkout main
git checkout -b task-06-captioning
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_captioning.py`:

```python
from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from aircraft_damage.captioning import (
    CAPTION_PROMPT,
    SUMMARY_PROMPT,
    BlipDescriber,
    ImageDescription,
)


class StubProcessor:
    """Records the prompts it is asked to encode and returns canned decodings."""

    def __init__(self, decoded: list[str]) -> None:
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_captioning.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'aircraft_damage.captioning'`

- [ ] **Step 4: Write `src/aircraft_damage/captioning.py`**

```python
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

from aircraft_damage import tf_env  # noqa: F401  # must precede the tensorflow import

import tensorflow as tf  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import BlipForConditionalGeneration, BlipProcessor  # noqa: E402

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
    def load(cls, model_id: str) -> BlipDescriber:
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
```

Note the one behavioural change from the notebook: `process_image` no longer wraps
everything in `except Exception: return "Error processing image"`. Swallowing every
error and returning a string that looks like a caption is exactly the failure mode
convention 6 exists to prevent — the Streamlit layer catches and renders errors instead.

- [ ] **Step 5: Run the fast tests to verify they pass**

Run: `uv run pytest tests/test_captioning.py -v -m "not slow"`
Expected: PASS, 4 tests.

- [ ] **Step 6: Run the slow test, which downloads BLIP**

Run: `uv run pytest tests/test_captioning.py -v -m slow`
Expected: PASS, 1 test. First run downloads ~1 GB into `~/.cache/huggingface`; allow
several minutes.

- [ ] **Step 7: Run the full check script**

Run: `./scripts/check.sh`
Expected: `All checks passed.`

- [ ] **Step 8: Commit and merge the task branch**

```bash
git add -A
git commit -m "feat: add BLIP describer and the notebook's custom captioning layer"
git checkout main
git merge --no-ff task-06-captioning -m "merge: task 06 BLIP captioning"
```

---

## Task 7: Ollama in Docker and the LLM client

**Files:**
- Create: `docker-compose.yml`, `src/aircraft_damage/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `Config.ollama_host`, `Config.ollama_model`, `Config.ollama_timeout_s` (Task 1).
- Produces: `OllamaError(RuntimeError)`, `strip_thinking(text: str) -> str`, and `OllamaClient` with `__init__(host: str, model: str, timeout_s: int = 180, client: httpx.Client | None = None)`, `is_available() -> bool`, `available_models() -> list[str]`, `has_model() -> bool`, and `chat(system: str, user: str, *, temperature: float = 0.2) -> str`. Task 8 calls `chat`; Task 10 calls `is_available` and `has_model`.

- [ ] **Step 1: Create the task branch**

```bash
git checkout main
git checkout -b task-07-ollama
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: adc-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-models:/root/.ollama
    environment:
      # This host has 6 GB of RAM and no GPU. Keep exactly one model resident,
      # serve one request at a time, and unload after five idle minutes.
      OLLAMA_KEEP_ALIVE: "5m"
      OLLAMA_MAX_LOADED_MODELS: "1"
      OLLAMA_NUM_PARALLEL: "1"
    mem_limit: 4g
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 20s

volumes:
  ollama-models:
```

- [ ] **Step 3: Start Ollama and pull the model**

```bash
docker compose up -d
```

Wait for health, then:

```bash
docker compose exec ollama ollama pull qwen3:4b
```

Expected: ~2.6 GB downloaded, ending in `success`. Verify:

```bash
curl -s http://localhost:11434/api/tags | head -c 400
```

Expected: JSON containing `"name":"qwen3:4b"`.

Once it works, pin the image for reproducibility — read the resolved digest and
replace `ollama/ollama:latest` in `docker-compose.yml` with the concrete version tag:

```bash
docker compose exec ollama ollama --version
```

- [ ] **Step 4: Write the failing test**

Create `tests/test_llm.py`:

```python
from __future__ import annotations

import httpx
import pytest

from aircraft_damage.llm import OllamaClient, OllamaError, strip_thinking

HOST = "http://ollama.test:11434"


def _client(handler: object) -> OllamaClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return OllamaClient(HOST, "qwen3:4b", client=httpx.Client(transport=transport))


def test_strip_thinking_removes_a_reasoning_block() -> None:
    assert strip_thinking("<think>weighing options</think>\n\n## Finding\nA dent.") == (
        "## Finding\nA dent."
    )


def test_strip_thinking_handles_multiline_blocks() -> None:
    text = "<think>\nline one\nline two\n</think>Result"

    assert strip_thinking(text) == "Result"


def test_strip_thinking_leaves_ordinary_text_alone() -> None:
    assert strip_thinking("## Finding\nA crack.") == "## Finding\nA crack."


def test_chat_posts_the_configured_model_and_messages() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = httpx.Request("POST", "http://x", content=request.content).content
        import json

        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "A report."}})

    result = _client(handler).chat("You are an inspector.", "Evidence here.")

    payload = seen["payload"]
    assert seen["url"] == f"{HOST}/api/chat"
    assert payload["model"] == "qwen3:4b"  # type: ignore[index]
    assert payload["stream"] is False  # type: ignore[index]
    assert payload["messages"] == [  # type: ignore[index]
        {"role": "system", "content": "You are an inspector."},
        {"role": "user", "content": "Evidence here."},
    ]
    assert result == "A report."


def test_chat_strips_inline_thinking_from_the_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "<think>hmm</think>Done."}})

    assert _client(handler).chat("sys", "user") == "Done."


def test_chat_retries_without_the_think_field_when_the_server_rejects_it() -> None:
    attempts: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)
        attempts.append("think" in payload)
        if "think" in payload:
            return httpx.Response(400, text="unknown field think")
        return httpx.Response(200, json={"message": {"content": "Done."}})

    assert _client(handler).chat("sys", "user") == "Done."
    assert attempts == [True, False]


def test_chat_raises_on_a_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="model runner crashed")

    with pytest.raises(OllamaError, match="500"):
        _client(handler).chat("sys", "user")


def test_chat_raises_on_an_empty_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "   "}})

    with pytest.raises(OllamaError, match="empty"):
        _client(handler).chat("sys", "user")


def test_chat_raises_when_the_host_is_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(OllamaError, match="Could not reach Ollama"):
        _client(handler).chat("sys", "user")


def test_is_available_is_true_when_tags_responds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    assert _client(handler).is_available() is True


def test_is_available_is_false_when_the_host_refuses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    assert _client(handler).is_available() is False


def test_has_model_matches_the_configured_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:4b"}, {"name": "llama3:8b"}]})

    client = _client(handler)
    assert client.available_models() == ["qwen3:4b", "llama3:8b"]
    assert client.has_model() is True
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aircraft_damage.llm'`

- [ ] **Step 6: Write `src/aircraft_damage/llm.py`**

```python
"""HTTP client for a local Ollama server.

The model this talks to (``qwen3:4b``) has no vision encoder. Nothing in this
module accepts or transmits image data — only text.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CHAT_ENDPOINT = "/api/chat"
TAGS_ENDPOINT = "/api/tags"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT_S = 180
PROBE_TIMEOUT_S = 3.0
KEEP_ALIVE = "5m"
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
THINKING_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns something unusable."""


def strip_thinking(text: str) -> str:
    """Remove inline ``<think>...</think>`` reasoning blocks.

    Ollama versions that surface Qwen3's reasoning in a separate ``thinking``
    field need no help here, but older builds inline it into the content.

    Args:
        text: Raw assistant content.

    Returns:
        The content with reasoning blocks removed and surrounding space trimmed.
    """
    return THINKING_PATTERN.sub("", text).strip()


class OllamaClient:
    """Talks to an Ollama server over HTTP."""

    def __init__(
        self,
        host: str,
        model: str,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialise the client.

        Args:
            host: Base URL of the Ollama server, e.g. ``http://localhost:11434``.
            model: Model tag to generate with, e.g. ``qwen3:4b``.
            timeout_s: Request timeout for generation calls.
            client: An ``httpx.Client`` to reuse; tests inject a mock transport.
        """
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._client = client if client is not None else httpx.Client(timeout=timeout_s)

    def is_available(self) -> bool:
        """Report whether the server answers at all.

        Returns:
            ``True`` if ``/api/tags`` responds with 200.
        """
        try:
            response = self._client.get(f"{self.host}{TAGS_ENDPOINT}", timeout=PROBE_TIMEOUT_S)
        except httpx.HTTPError:
            return False
        return response.status_code == HTTP_OK

    def available_models(self) -> list[str]:
        """List the model tags the server has pulled.

        Returns:
            Model names, or an empty list if the server cannot be reached.
        """
        try:
            response = self._client.get(f"{self.host}{TAGS_ENDPOINT}", timeout=PROBE_TIMEOUT_S)
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        models: list[dict[str, Any]] = response.json().get("models", [])
        return [str(entry["name"]) for entry in models]

    def has_model(self) -> bool:
        """Report whether the configured model has been pulled.

        Returns:
            ``True`` if ``self.model`` appears in the server's tag list.
        """
        return self.model in self.available_models()

    def chat(self, system: str, user: str, *, temperature: float = DEFAULT_TEMPERATURE) -> str:
        """Send a system/user pair and return the assistant's text.

        Args:
            system: The system prompt.
            user: The user message.
            temperature: Sampling temperature.

        Returns:
            The assistant's content, with reasoning blocks stripped.

        Raises:
            OllamaError: If the server is unreachable, errors, or returns nothing.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Qwen3 is a hybrid reasoning model; on a CPU-only host the thinking
            # pass costs far more than it buys for this templated report.
            "think": False,
            "options": {"temperature": temperature},
            "keep_alive": KEEP_ALIVE,
        }

        response = self._post_chat(payload)
        if response.status_code == HTTP_BAD_REQUEST:
            logger.warning("Ollama rejected the 'think' field; retrying without it")
            payload.pop("think")
            response = self._post_chat(payload)

        if response.status_code != HTTP_OK:
            raise OllamaError(f"Ollama returned {response.status_code}: {response.text[:200]}")

        content = str(response.json().get("message", {}).get("content", ""))
        cleaned = strip_thinking(content)
        if not cleaned:
            raise OllamaError(f"Ollama returned an empty response from {self.model}")
        return cleaned

    def _post_chat(self, payload: dict[str, Any]) -> httpx.Response:
        """POST to the chat endpoint, translating transport errors.

        Args:
            payload: The JSON request body.

        Returns:
            The raw response, whatever its status code.

        Raises:
            OllamaError: If the request could not be delivered.
        """
        try:
            return self._client.post(
                f"{self.host}{CHAT_ENDPOINT}",
                json=payload,
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.host}: {exc}") from exc
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `uv run pytest tests/test_llm.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 8: Smoke-test against the real server**

```bash
uv run python -c "
from aircraft_damage.config import load_config
from aircraft_damage.llm import OllamaClient

config = load_config()
client = OllamaClient(config.ollama_host, config.ollama_model, config.ollama_timeout_s)
print('available:', client.is_available())
print('models:', client.available_models())
print(client.chat('Answer in one short sentence.', 'What is metal fatigue?'))
"
```

Expected: `available: True`, a list containing `qwen3:4b`, and one sentence of prose.
First call is slow (model load, ~30–60 s on CPU); later calls are faster while the
model stays resident.

- [ ] **Step 9: Run the full check script**

Run: `./scripts/check.sh`
Expected: `All checks passed.`

- [ ] **Step 10: Commit and merge the task branch**

```bash
git add -A
git commit -m "feat: add dockerised Ollama service and typed HTTP client"
git checkout main
git merge --no-ff task-07-ollama -m "merge: task 07 Ollama service and client"
```

---

## Task 8: Evidence packet and report generation

**Files:**
- Create: `src/aircraft_damage/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `OllamaClient.chat` (Task 7).
- Produces: `EvidencePacket` (frozen dataclass: `filename: str`, `image_size: tuple[int, int]`, `predicted_label: str`, `confidence: float`, `probabilities: dict[str, float]`, `caption: str`, `summary: str`, `model_test_accuracy: float | None`), `SYSTEM_PROMPT: str`, `REPORT_SECTIONS: tuple[str, ...]`, `build_user_prompt(packet: EvidencePacket) -> str`, and `generate_report(packet: EvidencePacket, client: OllamaClient, *, temperature: float = 0.2) -> str`. Tasks 9 and 10 use all of these.

- [ ] **Step 1: Create the task branch**

```bash
git checkout main
git checkout -b task-08-report
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_report.py`:

```python
from __future__ import annotations

from aircraft_damage.report import (
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aircraft_damage.report'`

- [ ] **Step 4: Write `src/aircraft_damage/report.py`**

```python
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
        f"{packet.model_test_accuracy:.4f}"
        if packet.model_test_accuracy is not None
        else "unknown"
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 6: Eyeball a real report**

```bash
uv run python -c "
from aircraft_damage.config import load_config
from aircraft_damage.llm import OllamaClient
from aircraft_damage.report import EvidencePacket, generate_report

config = load_config()
client = OllamaClient(config.ollama_host, config.ollama_model, config.ollama_timeout_s)
packet = EvidencePacket(
    filename='149_22.jpg',
    image_size=(640, 480),
    predicted_label='dent',
    confidence=0.8734,
    probabilities={'crack': 0.1266, 'dent': 0.8734},
    caption='a close up of a metal surface with a dent',
    summary='a detailed photo showing a dented panel on an aircraft wing',
    model_test_accuracy=0.88,
)
print(generate_report(packet, client))
"
```

Expected: Markdown containing all six `##` sections. **Read it.** Confirm no sentence
claims the model looked at the image. If any does, tighten rule 2 in `SYSTEM_PROMPT`
and re-run — do not proceed with a prompt that produces false claims.

- [ ] **Step 7: Run the full check script**

Run: `./scripts/check.sh`
Expected: `All checks passed.`

- [ ] **Step 8: Commit and merge the task branch**

```bash
git add -A
git commit -m "feat: add evidence packet and LLM report generation"
git checkout main
git merge --no-ff task-08-report -m "merge: task 08 report generation"
```

---

## Task 9: Inspection pipeline

**Files:**
- Create: `src/aircraft_damage/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `DamageClassifier.predict` / `ClassificationResult` (Task 5), `BlipDescriber.describe` / `ImageDescription` (Task 6), `OllamaClient` (Task 7), `EvidencePacket` / `generate_report` (Task 8).
- Produces: `InspectionResult` (frozen dataclass: `packet: EvidencePacket`, `report: str`), `build_packet(image, filename, classification, description, test_accuracy) -> EvidencePacket`, and `run_inspection(image, filename, *, classifier, describer, client, test_accuracy=None, temperature=0.2) -> InspectionResult`. Task 10's UI calls `build_packet` directly so it can render each stage as it completes.

- [ ] **Step 1: Create the task branch**

```bash
git checkout main
git checkout -b task-09-pipeline
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_pipeline.py`:

```python
from __future__ import annotations

from PIL import Image

from aircraft_damage.captioning import ImageDescription
from aircraft_damage.classifier import ClassificationResult
from aircraft_damage.pipeline import build_packet, run_inspection

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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aircraft_damage.pipeline'`

- [ ] **Step 4: Write `src/aircraft_damage/pipeline.py`**

```python
"""End-to-end inspection: classify, describe, then report.

This module owns the ordering and the data flow. The Streamlit app calls the
individual pieces so it can render progress, but the composed
:func:`run_inspection` is the tested reference path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PIL import Image

from aircraft_damage.captioning import BlipDescriber, ImageDescription
from aircraft_damage.classifier import ClassificationResult, DamageClassifier
from aircraft_damage.llm import OllamaClient
from aircraft_damage.report import DEFAULT_TEMPERATURE, EvidencePacket, generate_report

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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 6: Run the full check script**

Run: `./scripts/check.sh`
Expected: `All checks passed.`

- [ ] **Step 7: Commit and merge the task branch**

```bash
git add -A
git commit -m "feat: compose classification, description, and reporting into a pipeline"
git checkout main
git merge --no-ff task-09-pipeline -m "merge: task 09 inspection pipeline"
```

---

## Task 10: Streamlit app, README, and end-to-end verification

**Files:**
- Create: `src/aircraft_damage/app/__init__.py`, `src/aircraft_damage/app/styles.py`, `src/aircraft_damage/app/streamlit_app.py`, `.streamlit/config.toml`, `README.md`

**Interfaces:**
- Consumes: `load_config` (Task 1), `DamageClassifier.load` / `ModelNotTrainedError` (Task 5), `BlipDescriber.load` (Task 6), `OllamaClient` / `OllamaError` (Task 7), `generate_report` (Task 8), `build_packet` (Task 9).
- Produces: nothing consumed by other tasks. This layer holds no domain logic — anything worth testing belongs one module down.

- [ ] **Step 1: Create the task branch**

```bash
git checkout main
git checkout -b task-10-app
```

- [ ] **Step 2: Write `.streamlit/config.toml`**

```toml
[theme]
base = "dark"
primaryColor = "#4F8DF7"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#161A23"
textColor = "#E6E9EF"
font = "sans serif"

[server]
maxUploadSize = 10

[browser]
gatherUsageStats = false
```

- [ ] **Step 3: Write `src/aircraft_damage/app/__init__.py`**

```python
"""Streamlit interface for the aircraft damage inspection pipeline."""

from __future__ import annotations
```

- [ ] **Step 4: Write `src/aircraft_damage/app/styles.py`**

```python
"""Presentation-only CSS for the Streamlit app."""

from __future__ import annotations

CUSTOM_CSS = """
<style>
  .block-container { padding-top: 2.5rem; max-width: 1200px; }

  .adc-hero {
    background: linear-gradient(135deg, #1B2233 0%, #0E1117 60%);
    border: 1px solid #232A3A;
    border-radius: 16px;
    padding: 1.6rem 1.9rem;
    margin-bottom: 1.6rem;
  }
  .adc-hero h1 {
    font-size: 1.75rem;
    font-weight: 650;
    margin: 0 0 .35rem 0;
    letter-spacing: -0.01em;
  }
  .adc-hero p { color: #9AA4B8; margin: 0; font-size: .95rem; }

  .adc-pill {
    display: inline-block;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    padding: .22rem .6rem;
    border-radius: 999px;
    margin-right: .4rem;
  }
  .adc-pill-crack { background: #3A1D22; color: #FF8A8A; border: 1px solid #5A2A32; }
  .adc-pill-dent  { background: #1D2E3A; color: #7CC7FF; border: 1px solid #2A4A5A; }

  .adc-report {
    background: #12161F;
    border: 1px solid #232A3A;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
  }
  .adc-report h2 {
    font-size: 1.02rem;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: #7CC7FF;
    border-bottom: 1px solid #232A3A;
    padding-bottom: .35rem;
    margin-top: 1.4rem;
  }
  .adc-report h2:first-child { margin-top: 0; }

  .adc-note {
    color: #7A8296;
    font-size: .82rem;
    border-left: 2px solid #2A3245;
    padding-left: .7rem;
    margin-top: 1rem;
  }

  div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
"""
```

- [ ] **Step 5: Write `src/aircraft_damage/app/streamlit_app.py`**

```python
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
            st.code(f"docker compose exec ollama ollama pull {config.ollama_model}", language="bash")
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
        st.image(image, caption=uploaded.name, use_container_width=True)

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
```

Note on the report container: `st.container(border=True)` is used rather than wrapping
the Markdown in a `<div class="adc-report">`. Streamlit escapes HTML inside
`st.markdown` unless `unsafe_allow_html=True`, and turning that on for
model-generated text would let the LLM inject markup into the page. The `.adc-report`
rule in `styles.py` therefore targets Streamlit's own bordered container — change its
selector to `div[data-testid="stVerticalBlockBorderWrapper"]` if you want the custom
heading colours to apply, or leave the default border as-is.

- [ ] **Step 6: Run the app**

```bash
uv run streamlit run src/aircraft_damage/app/streamlit_app.py
```

Expected: opens on `http://localhost:8501` with the sidebar showing two green
"ready" indicators. If Ollama shows red, run `docker compose up -d` and reload.

- [ ] **Step 7: End-to-end verification**

Upload a test-set image, e.g.
`data/aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg`.

Verify each of these, and fix before moving on if any fails:

- [ ] The image preview renders.
- [ ] The status panel shows all three stages and ends in "complete".
- [ ] A label and a confidence metric appear, plus the probability bar chart.
- [ ] The expander shows a non-empty BLIP caption and description.
- [ ] The report renders with all six `## ` sections.
- [ ] **Read the report.** No sentence claims the model looked at the image.
- [ ] "Regenerate report" produces a new report without re-running classification
      (the status panel for the first two stages does not reappear).
- [ ] "Download report" saves a `.md` file with the report text.
- [ ] Upload a second, different image: the analysis re-runs for the new file.
- [ ] Stop Ollama (`docker compose stop`), reload, upload: the sidebar turns red with
      the `docker compose up -d` command, and the report stage shows a clean
      `st.error`, not a traceback. Then `docker compose start`.

- [ ] **Step 8: Write `README.md`**

```markdown
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

## Train

```bash
uv run python -m aircraft_damage.train
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
| First report takes ~60 s | Ollama is loading the model into RAM. Later calls reuse it for 5 minutes. |
| Training killed by the OOM reaper | Close other apps; this host has 6 GB total and Ollama holds up to 4 GB. `docker compose stop` while training. |
| `tensorflow-cpu` will not install | The venv is not Python 3.12. Recreate it with `uv venv --python 3.12`. |

## Limitations

- Two classes only; an undamaged surface is still forced into `crack` or `dent`.
- Trained on 300 images. Treat confidence numbers as indicative, not calibrated.
- BLIP-base is a general-purpose captioner with no aircraft-specific training.
- Output is a triage aid, never an airworthiness determination.
```

- [ ] **Step 9: Run the full check script**

Run: `./scripts/check.sh`
Expected: `All checks passed.`

- [ ] **Step 10: Commit and merge the task branch**

```bash
git add -A
git commit -m "feat: add Streamlit inspection app and project README"
git checkout main
git merge --no-ff task-10-app -m "merge: task 10 Streamlit app and documentation"
```

- [ ] **Step 11: Confirm the branch history**

```bash
git log --oneline --graph --all | head -40
git branch
```

Expected: ten merge commits on `main`, and ten `task-NN-*` branches still present.

---

## Plan self-review

Checked against `docs/superpowers/specs/2026-08-21-aircraft-damage-classification.md`:

**Spec coverage.** Every spec section maps to a task: notebook 1.1–1.2 → Task 2;
1.3 → Task 3; 1.4–1.6 → Task 4; 1.7 and Part 2 → Tasks 5 and 6; the LLM stage → Tasks
7 and 8; the Streamlit requirements → Task 10; the artifacts contract → Task 4's
`save_metrics`; the text-only rule → Task 8's `SYSTEM_PROMPT` plus Task 10's sidebar
caption and footer note; the RAM constraints → Task 7's compose limits and Task 10's
lazy `@st.cache_resource` loaders. The Python conventions the user asked for are in
Global Constraints, mechanised in Task 1's `pyproject.toml` and `scripts/check.sh`,
and re-stated in the project's `CLAUDE.md`. The branch-per-task requirement appears in
Global Constraints and as the first and last steps of every task.

**Type consistency.** `class_indices` is name→index everywhere (`metrics.json`,
`build_generators`); `class_names` is index→name everywhere (`DamageClassifier`),
converted only by `class_names_from_indices`. `ClassificationResult.confidence` is
always the probability of the *predicted* label; `probabilities` is always keyed by
class name. `EvidencePacket.image_size` is `(width, height)` throughout, matching
`PIL.Image.size`. `build_model` and `build_feature_extractor` both take `weights` and
both default to `"imagenet"`.

**One known rough edge, flagged rather than hidden.** Task 3's
`test_model_has_the_notebook_topology` asserts the first layer's type name is
`"Functional"`. Keras 3 has renamed nested-model wrappers before; if it fails, the
step tells the implementer to adjust the *expected string*, never the topology.

**One security note carried into the code.** The report is model-generated text
rendered into a web page. It goes through `st.markdown` with `unsafe_allow_html`
left at its default of `False`, so the LLM cannot inject markup or script into the
app. Only the app's own static CSS and hero block use `unsafe_allow_html=True`.
