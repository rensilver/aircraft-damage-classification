from __future__ import annotations

from pathlib import Path

import pytest

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.test:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")

    config = load_config()

    assert config.ollama_host == "http://ollama.test:11434"
    assert config.ollama_model == "qwen3:8b"


def test_load_config_falls_back_to_local_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    config = load_config()

    assert config.ollama_host == "http://localhost:11434"
    assert config.ollama_model == "qwen3:4b"


def test_ollama_timeout_defaults_to_600_seconds(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")

    assert config.ollama_timeout_s == 600
