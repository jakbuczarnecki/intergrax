# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from local_workspace_application.benchmarks.local_model_qualification.config import load_config

_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "local-model-qualification.toml"
)
_APP_ROOT = Path(__file__).resolve().parents[2]


def test_valid_committed_config_loads() -> None:
    config = load_config(_CONFIG)
    assert config.schema_version == 1
    assert len(config.models) == 5


def test_paths_resolve_relative_to_toml() -> None:
    config = load_config(_CONFIG)
    assert config.results_json_path == (
        _APP_ROOT / "benchmarks/local_model_qualification/results/latest.json"
    ).resolve()
    assert config.report_markdown_path == (_APP_ROOT / "docs/LOCAL_MODEL_QUALIFICATION.md").resolve()


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    content = _CONFIG.read_text(encoding="utf-8") + "\nunknown = true\n"
    path = tmp_path / "bad.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(Exception):
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
