# Aircraft Damage Classification — Conventions

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
