# Improving Project Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 12 flat modules in `src/aircraft_damage/` into domain
subpackages (`vision/`, `reporting/`) so files are grouped by field of
action, matching the pattern real Python projects use, without changing any
runtime behavior.

**Architecture:** Pure move-and-rewire. No function, class, or behavior
changes. `config.py`, `tf_env.py`, and `pipeline.py` stay at package root
because they are cross-cutting (imported by both new subpackages and by
`app/`). Vision-only modules (`data.py`, `model.py`, `classifier.py`,
`captioning.py`, `train.py`) move to `vision/`. LLM/report modules
(`llm.py`, `report.py`) move to `reporting/`. `app/` is already its own
package and is untouched except for import paths. Every moved file keeps
its filename, so bare-name references in `CLAUDE.md` (e.g. `llm.strip_thinking`)
stay accurate.

**Tech Stack:** Python 3.12, existing pytest/ruff/mypy toolchain — no new
dependencies.

**Spec:** This plan is scoped directly from the user's request (no separate
spec doc): reorganize `src/` into folders by field of action, fix imports,
work on branch `refactor/improving-project-structure`.

## Global Constraints

- No behavior changes — this is a pure file-move refactor. Existing tests
  must pass unmodified in content (only their import paths change).
- Every moved/new module keeps `from __future__ import annotations` and a
  Google-style module docstring per `CLAUDE.md` conventions.
- `config.py` stays stdlib-only and at package root (`CLAUDE.md` invariant).
- `aircraft_damage.tf_env` stays importable at its current path — vision
  modules import it before `tensorflow`/`keras` (`CLAUDE.md` invariant).
- `ruff check`, `ruff format --check`, `mypy`, and `pytest -m "not slow"`
  (`./scripts/check.sh`) must all pass at the end.

---

## Target layout

```
src/aircraft_damage/
  __init__.py
  config.py
  tf_env.py
  pipeline.py
  vision/
    __init__.py
    data.py
    model.py
    classifier.py
    captioning.py
    train.py
  reporting/
    __init__.py
    llm.py
    report.py
  app/
    __init__.py
    streamlit_app.py
    styles.py
```

### Task 1: Create the `vision` subpackage

**Files:**
- Create: `src/aircraft_damage/vision/__init__.py`
- Move: `src/aircraft_damage/data.py` → `src/aircraft_damage/vision/data.py`
- Move: `src/aircraft_damage/model.py` → `src/aircraft_damage/vision/model.py`
- Move: `src/aircraft_damage/classifier.py` → `src/aircraft_damage/vision/classifier.py`
- Move: `src/aircraft_damage/captioning.py` → `src/aircraft_damage/vision/captioning.py`
- Move: `src/aircraft_damage/train.py` → `src/aircraft_damage/vision/train.py`
- Modify: `src/aircraft_damage/vision/train.py` (import paths + docstring)
- Modify: `src/aircraft_damage/vision/classifier.py` (error message text)
- Test: `tests/test_data.py`, `tests/test_model.py`, `tests/test_classifier.py`,
  `tests/test_captioning.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: nothing new — `aircraft_damage.tf_env` and `aircraft_damage.config`
  stay at their current import paths.
- Produces: `aircraft_damage.vision.data.build_generators`,
  `aircraft_damage.vision.model.build_model`,
  `aircraft_damage.vision.model.build_feature_extractor`,
  `aircraft_damage.vision.classifier.DamageClassifier`,
  `aircraft_damage.vision.classifier.ClassificationResult`,
  `aircraft_damage.vision.classifier.ModelNotTrainedError`,
  `aircraft_damage.vision.classifier.class_names_from_indices`,
  `aircraft_damage.vision.captioning.BlipDescriber`,
  `aircraft_damage.vision.captioning.ImageDescription`,
  `aircraft_damage.vision.train.plot_curves`,
  `aircraft_damage.vision.train.run_training`,
  `aircraft_damage.vision.train.save_metrics`,
  `aircraft_damage.vision.train.set_seeds` — these are what Task 3 (pipeline/app)
  and the test files import.

- [ ] **Step 1: Create the package and move the files**

```bash
mkdir -p src/aircraft_damage/vision
git mv src/aircraft_damage/data.py src/aircraft_damage/vision/data.py
git mv src/aircraft_damage/model.py src/aircraft_damage/vision/model.py
git mv src/aircraft_damage/classifier.py src/aircraft_damage/vision/classifier.py
git mv src/aircraft_damage/captioning.py src/aircraft_damage/vision/captioning.py
git mv src/aircraft_damage/train.py src/aircraft_damage/vision/train.py
```

Create `src/aircraft_damage/vision/__init__.py`:

```python
"""Data loading, model building, training, inference, and captioning."""

from __future__ import annotations
```

- [ ] **Step 2: Fix `vision/train.py`'s internal imports and docstring**

In `src/aircraft_damage/vision/train.py`, the module docstring's usage
example and the two intra-package imports change (only the `data`/`model`
lines — `tf_env` and `config` are unaffected since they're still at
`aircraft_damage.tf_env` / `aircraft_damage.config`):

```python
from aircraft_damage.vision.data import build_generators  # noqa: E402
from aircraft_damage.vision.model import build_model  # noqa: E402
```

And update the module-level docstring/CLI usage line from
`uv run python -m aircraft_damage.train` to
`uv run python -m aircraft_damage.vision.train`.

- [ ] **Step 3: Fix `vision/classifier.py`'s error message**

Both occurrences of `"Run 'uv run python -m aircraft_damage.train' first."`
become `"Run 'uv run python -m aircraft_damage.vision.train' first."`.

- [ ] **Step 4: Fix the moved tests' import paths**

```python
# tests/test_data.py
from aircraft_damage.vision.data import build_generators

# tests/test_model.py
from aircraft_damage.vision.model import build_feature_extractor, build_model

# tests/test_classifier.py
from aircraft_damage.vision.classifier import (
    ...  # same names, new module path
)

# tests/test_captioning.py
from aircraft_damage.vision.captioning import (
    ...  # same names, new module path
)

# tests/test_train.py
from aircraft_damage.vision.train import plot_curves, run_training, save_metrics, set_seeds
```

(Keep whatever else each import statement already lists — only the module
path segment changes from `aircraft_damage.X` to `aircraft_damage.vision.X`.)

- [ ] **Step 5: Run the moved tests**

Run: `uv run pytest tests/test_data.py tests/test_model.py tests/test_classifier.py tests/test_captioning.py tests/test_train.py -m "not slow" -v`
Expected: PASS (import errors would show as collection failures — fix any
remaining `aircraft_damage.data` / `.model` / `.classifier` / `.captioning`
/ `.train` references they surface).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: move vision modules into aircraft_damage.vision"
```

---

### Task 2: Create the `reporting` subpackage

**Files:**
- Create: `src/aircraft_damage/reporting/__init__.py`
- Move: `src/aircraft_damage/llm.py` → `src/aircraft_damage/reporting/llm.py`
- Move: `src/aircraft_damage/report.py` → `src/aircraft_damage/reporting/report.py`
- Modify: `src/aircraft_damage/reporting/report.py` (import of `llm`)
- Test: `tests/test_llm.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `aircraft_damage.reporting.llm.OllamaClient`,
  `aircraft_damage.reporting.llm.OllamaError`,
  `aircraft_damage.reporting.llm.strip_thinking`,
  `aircraft_damage.reporting.report.generate_report`,
  `aircraft_damage.reporting.report.build_user_prompt`,
  `aircraft_damage.reporting.report.EvidencePacket`,
  `aircraft_damage.reporting.report.DEFAULT_TEMPERATURE`,
  `aircraft_damage.reporting.report.LOW_CONFIDENCE_THRESHOLD` — consumed by
  Task 3.

- [ ] **Step 1: Create the package and move the files**

```bash
mkdir -p src/aircraft_damage/reporting
git mv src/aircraft_damage/llm.py src/aircraft_damage/reporting/llm.py
git mv src/aircraft_damage/report.py src/aircraft_damage/reporting/report.py
```

Create `src/aircraft_damage/reporting/__init__.py`:

```python
"""Ollama client and maintenance-report generation."""

from __future__ import annotations
```

- [ ] **Step 2: Fix `reporting/report.py`'s import**

```python
from aircraft_damage.reporting.llm import OllamaClient
```

- [ ] **Step 3: Fix the moved tests' import paths**

```python
# tests/test_llm.py
from aircraft_damage.reporting.llm import OllamaClient, OllamaError, strip_thinking

# tests/test_report.py
from aircraft_damage.reporting.report import (
    ...  # same names, new module path
)
```

- [ ] **Step 4: Run the moved tests**

Run: `uv run pytest tests/test_llm.py tests/test_report.py -m "not slow" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move llm and report modules into aircraft_damage.reporting"
```

---

### Task 3: Rewire `pipeline.py`, `app/streamlit_app.py`, and remaining tests

**Files:**
- Modify: `src/aircraft_damage/pipeline.py`
- Modify: `src/aircraft_damage/app/streamlit_app.py`
- Modify: `tests/test_pipeline.py`
- Test: `tests/test_pipeline.py`, full suite

**Interfaces:**
- Consumes: everything produced in Task 1 and Task 2.
- Produces: nothing new — `aircraft_damage.pipeline.build_packet` and
  `aircraft_damage.pipeline.run_inspection` keep their existing path (this
  module does not move).

- [ ] **Step 1: Fix `pipeline.py`'s imports**

```python
from aircraft_damage.vision.classifier import ClassificationResult, DamageClassifier
from aircraft_damage.reporting.llm import OllamaClient
from aircraft_damage.reporting.report import DEFAULT_TEMPERATURE, EvidencePacket, generate_report
```

And the lazy import inside the function body:

```python
from aircraft_damage.vision.captioning import BlipDescriber, ImageDescription
```

- [ ] **Step 2: Fix `app/streamlit_app.py`'s imports**

```python
from aircraft_damage.app.styles import CUSTOM_CSS
from aircraft_damage.vision.classifier import DamageClassifier, ModelNotTrainedError
from aircraft_damage.config import Config, load_config
from aircraft_damage.reporting.llm import OllamaClient, OllamaError
from aircraft_damage.pipeline import build_packet
from aircraft_damage.reporting.report import LOW_CONFIDENCE_THRESHOLD, generate_report
```

(Final ordering will be fixed by `ruff format`/`ruff check --fix` in Step 4
below — isort groups these alphabetically within the first-party block.)

The lazy import:

```python
from aircraft_damage.vision.captioning import BlipDescriber  # noqa: PLC0415
```

And the training-command hint shown in the UI:

```python
st.code("uv run python -m aircraft_damage.vision.train", language="bash")
```

- [ ] **Step 3: Fix `tests/test_pipeline.py`'s imports**

```python
from aircraft_damage.vision.captioning import ImageDescription
from aircraft_damage.vision.classifier import ClassificationResult
from aircraft_damage.pipeline import build_packet, run_inspection
```

- [ ] **Step 4: Auto-fix import order, then run the full check**

```bash
uv run ruff check --fix src tests
uv run ruff format src tests
./scripts/check.sh
```

Expected: all four checks (`ruff check`, `ruff format --check`, `mypy`,
`pytest -m "not slow"`) pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: rewire pipeline and streamlit app to new subpackages"
```

---

### Task 4: Update docs that reference the old module paths

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing — documentation only.

- [ ] **Step 1: Update `README.md`**

Line 54, `uv run python -m aircraft_damage.train` →
`uv run python -m aircraft_damage.vision.train`.

- [ ] **Step 2: Update `CLAUDE.md`**

Under `## Commands`, `Train: \`uv run python -m aircraft_damage.train\`` →
`Train: \`uv run python -m aircraft_damage.vision.train\``.

Leave `docs/superpowers/plans/2026-08-21-aircraft-damage-classification.md`
and `docs/superpowers/specs/2026-08-21-aircraft-damage-classification.md`
untouched — they are point-in-time records of prior work, not living docs.

- [ ] **Step 3: Verify no stale references remain**

Run: `grep -rn "aircraft_damage\.\(train\|llm\|report\|data\|model\|classifier\|captioning\)\b" --include="*.py" --include="*.md" README.md CLAUDE.md src tests`
Expected: every match uses the new `aircraft_damage.vision.*` /
`aircraft_damage.reporting.*` paths (or is inside `docs/superpowers/plans/`
or `docs/superpowers/specs/`, which are intentionally left alone).

- [ ] **Step 4: Final full check and commit**

```bash
./scripts/check.sh
git add README.md CLAUDE.md
git commit -m "docs: update commands for the new vision/reporting subpackages"
```
