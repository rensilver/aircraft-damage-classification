# Aircraft Damage Classification — Conventions

## Python coding conventions

1. **Minimal imports** — Don't import a module or function you don't use. When in doubt, comment why it's there.
2. **Explicit is better** — Prefer `if x is True` over `if x`, and `config.seed = 42` (direct assignment) over `setattr`.
3. **No type: ignore in tests** — Write the test so the type system passes. If you need `# type: ignore`, the test is incomplete or the code is wrong.
4. **No magic strings** — Hardcoded paths, URLs, regex patterns, or error messages beyond single-use cases belong in constants (module-level or class-level). Name them `SCREAMING_SNAKE_CASE`.
5. **Dataclasses, not namedtuples** — Use `@dataclass` for runtime behavior, including immutability with `frozen=True`. Namedtuples are for type-only code (rarely used here).
6. **Prefer positional args in signatures** — Only use `*args` or `**kwargs` for variadic cases or when forwarding to an external API. Named args are clearer.
7. **Avoid module-level side effects** — No `os.makedirs()`, no network calls, no randomness at import time. Exception: environment setup (e.g., `os.environ.setdefault`) in dedicated modules like `tf_env.py`.
8. **Tests import the public API** — Tests import from the module root (`from aircraft_damage.config import Config`), not from private implementation details. Exception: fixtures that set up or tear down private state.
9. **One assertion per test** (loosely) — When a test has many assertions, they should all check the same thing (e.g., properties of the same object). If you're testing unrelated concerns, split the test.
10. **Fixtures over setup/teardown** — Use pytest fixtures for state management, not `setUp` / `tearDown` methods (which are pytest-compatible but verbose).
11. **Monkeypatch, not manual mock** — Use `pytest.monkeypatch` for environment variables, not manual `os.environ` save/restore. Use `unittest.mock` only for complex patching (spies, call counts).
12. **Avoid pytest.approx() for non-floats** — Use it only for floating-point comparisons. For ints or strings, use exact equality.
13. **Descriptive test names** — Test names are documentation. `test_rejects_negative_seed` is clearer than `test_seed_validation`.
14. **No global mutable state** — Don't modify `sys.modules`, monkey-patch builtins, or share state between tests. Fixtures are the right way to share setup.
15. **One level of abstraction per function** — A function should not mix low-level details (regex, file I/O) with high-level logic (orchestration). Split them.

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
