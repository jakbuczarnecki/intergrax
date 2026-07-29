# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_workspace_application.benchmarks.local_model_qualification.config import load_config

_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "local-model-qualification.toml"
)
_APP_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "infra" / "docker" / "ollama" / "docker-compose.yml"


def test_valid_committed_config_loads() -> None:
    config = load_config(_CONFIG)
    assert config.schema_version == 2
    assert len(config.models) == 5
    assert sum(1 for model in config.models if model.enabled) == 5


def test_paths_resolve_relative_to_toml() -> None:
    config = load_config(_CONFIG)
    assert config.results_json_path == (
        _APP_ROOT / "benchmarks/local_model_qualification/results/latest.json"
    ).resolve()
    assert config.report_markdown_path == (_APP_ROOT / "docs/LOCAL_MODEL_QUALIFICATION.md").resolve()
    assert config.compose_file_path == _COMPOSE.resolve()


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8") + "\nunknown = true\n"
    path = tmp_path / "bad.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(Exception):
        load_config(path)


def test_pull_missing_models_rejected(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8").replace(
        'keep_alive = "10m"',
        'keep_alive = "10m"\npull_missing_models = false',
    )
    path = tmp_path / "pull.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_runtime_other_than_docker_rejected(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8").replace('runtime = "docker"', 'runtime = "native"')
    path = tmp_path / "runtime.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_compose_file_rejected(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8").replace(
        'compose_file = "../../../infra/docker/ollama/docker-compose.yml"',
        'compose_file = "../../../missing-compose.yml"',
    )
    path = tmp_path / "missing-compose.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="does not exist"):
        load_config(path)


def test_absolute_compose_file_rejected(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8").replace(
        'compose_file = "../../../infra/docker/ollama/docker-compose.yml"',
        'compose_file = "/tmp/docker-compose.yml"',
    )
    path = tmp_path / "abs-compose.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="relative"):
        load_config(path)


def test_compose_path_outside_repo_rejected(tmp_path: Path) -> None:
    outside = (_REPO_ROOT.parent / "outside-compose-qualification.yml").resolve()
    outside.write_text("services: {}\n", encoding="utf-8")
    try:
        rel = Path(os.path.relpath(outside, tmp_path))
        content = _CONFIG.read_text(encoding="utf-8").replace(
            'compose_file = "../../../infra/docker/ollama/docker-compose.yml"',
            f'compose_file = "{rel.as_posix()}"',
        )
        path = tmp_path / "escape-compose.toml"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="escapes"):
            load_config(path)
    finally:
        outside.unlink(missing_ok=True)


def test_startup_timeout_validation(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8").replace(
        "startup_timeout_seconds = 120",
        "startup_timeout_seconds = 0",
    )
    path = tmp_path / "startup.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_pull_timeout_validation(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8").replace(
        "model_pull_timeout_seconds = 7200",
        "model_pull_timeout_seconds = 0",
    )
    path = tmp_path / "pull-timeout.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_readiness_poll_validation(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8").replace(
        "readiness_poll_seconds = 1.0",
        "readiness_poll_seconds = 0",
    )
    path = tmp_path / "poll.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_duplicate_models_rejected(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8")
    content += '\n[[models]]\nname = "qwen2.5:14b"\nenabled = false\nrole = "dup"\n'
    path = tmp_path / "dup.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_config(path)


def test_no_enabled_model_rejected(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8").replace("enabled = true", "enabled = false")
    path = tmp_path / "disabled.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="enabled"):
        load_config(path)


def test_no_enabled_protocol_rejected(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8")
    content = content.replace("structured_output = true", "structured_output = false")
    content = content.replace("single_plan_tool = true", "single_plan_tool = false")
    path = tmp_path / "noproto.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        load_config(path)


def test_invalid_repetition_rejected(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8").replace("repetitions = 3", "repetitions = 0")
    path = tmp_path / "rep.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_invalid_threshold_rejected(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8").replace(
        "qualified_semantic_success_rate = 1.0",
        "qualified_semantic_success_rate = 1.5",
    )
    path = tmp_path / "thr.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_output_path_escaping_application_directory_rejected(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8").replace(
        'results_json = "../benchmarks/local_model_qualification/results/latest.json"',
        'results_json = "../../../escape.json"',
    )
    path = tmp_path / "escape.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_config(path)


def test_docker_compose_contract() -> None:
    text = _COMPOSE.read_text(encoding="utf-8")
    assert "ollama:" in text
    assert "container_name: intergrax-ollama" in text
    assert '"11434:11434"' in text or "'11434:11434'" in text or "11434:11434" in text
    assert "/root/.ollama" in text
    assert "intergrax-ollama-models" in text
    assert "restart: unless-stopped" in text
