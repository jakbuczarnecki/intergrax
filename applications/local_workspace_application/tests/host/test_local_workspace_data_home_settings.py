# © Artur Czarnecki. All rights reserved.

"""Tests for canonical LKW data-home settings contract (LKW.5A)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest

from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_DATA_HOME_ENV_KEYS = ("LOCAL_WORKSPACE_DATA_HOME", "LKW_DATA_HOME")
_DEFAULT_DATA_HOME = "build/local_workspace"
_ENV_EXAMPLE_PATH = Path(__file__).resolve().parents[2] / ".env.example"


@pytest.fixture(autouse=True)
def _clear_data_home_env() -> Generator[None, None, None]:
    saved = {key: os.environ[key] for key in _DATA_HOME_ENV_KEYS if key in os.environ}
    for key in _DATA_HOME_ENV_KEYS:
        os.environ.pop(key, None)
    yield
    for key in _DATA_HOME_ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in saved.items():
        os.environ[key] = value


def _posix(path: str) -> str:
    return Path(path).as_posix()


def test_data_home_defaults_to_repo_dev_build_path() -> None:
    settings = LocalWorkspaceBackendSettings.from_env()

    assert _posix(settings.data_home) == _DEFAULT_DATA_HOME
    assert _posix(settings.config_dir).endswith("config")
    assert _posix(settings.data_dir).endswith("data")
    assert _posix(settings.sqlite_data_dir).endswith("data/sqlite")
    assert _posix(settings.shadow_workspaces_dir).endswith("data/shadow_workspaces")
    assert _posix(settings.logs_dir).endswith("logs")
    assert _posix(settings.run_dir).endswith("run")


def test_local_workspace_data_home_env_takes_precedence() -> None:
    os.environ["LOCAL_WORKSPACE_DATA_HOME"] = "custom/lkw"
    os.environ["LKW_DATA_HOME"] = "legacy/lkw"

    settings = LocalWorkspaceBackendSettings.from_env()

    assert settings.data_home == "custom/lkw"


def test_lkw_data_home_alias_is_used_when_primary_missing() -> None:
    os.environ["LKW_DATA_HOME"] = "legacy/lkw"

    settings = LocalWorkspaceBackendSettings.from_env()

    assert settings.data_home == "legacy/lkw"


def test_blank_data_home_env_falls_back() -> None:
    os.environ["LOCAL_WORKSPACE_DATA_HOME"] = "   "

    settings = LocalWorkspaceBackendSettings.from_env()

    assert settings.data_home == _DEFAULT_DATA_HOME


def test_data_home_contract_does_not_change_rag_defaults() -> None:
    os.environ["LOCAL_WORKSPACE_DATA_HOME"] = "custom/lkw"

    settings = LocalWorkspaceBackendSettings.from_env()

    assert settings.enable_rag is True
    assert settings.enable_rag_ingest is True
    assert "rag.retrieve" in settings.enabled_tool_ids
    assert "rag.ingest_document" in settings.enabled_tool_ids


def test_env_example_documents_data_home_contract() -> None:
    content = _ENV_EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "LOCAL_WORKSPACE_DATA_HOME=build/local_workspace" in content
    assert "INTERGRAX_SQLITE_DATA_DIR" in content
    assert "INTERGRAX_SHADOW_ROOT" in content
    assert "LOCAL_WORKSPACE_VECTOR_STORE=qdrant" in content
