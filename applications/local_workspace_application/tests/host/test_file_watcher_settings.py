# © Artur Czarnecki. All rights reserved.

"""Tests for LKW file-watcher settings and sidecar configuration (LKW.7B2B)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Generator, cast

import pytest

from local_workspace_application.file_watcher.sidecar import (
    FileWatcherSidecarConfigurationError,
    build_file_watcher_sidecar_config,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _settings_from_env() -> LocalWorkspaceBackendSettings:
    return cast(
        LocalWorkspaceBackendSettings,
        LocalWorkspaceBackendSettings.from_env(),
    )


_WATCHER_ENV_KEYS = (
    "LOCAL_WORKSPACE_FILE_WATCHER_ENABLED",
    "LOCAL_WORKSPACE_FILE_WATCHER_TENANT_ID",
    "LOCAL_WORKSPACE_FILE_WATCHER_WORKSPACE_ID",
    "LOCAL_WORKSPACE_FILE_WATCHER_COLLECTION_ID",
    "LOCAL_WORKSPACE_FILE_WATCHER_POLL_INTERVAL_SECONDS",
    "LOCAL_WORKSPACE_FILE_WATCHER_DEBOUNCE_SECONDS",
    "LOCAL_WORKSPACE_FILE_WATCHER_MAX_BATCH_WAIT_SECONDS",
    "LOCAL_WORKSPACE_FILE_WATCHER_PRIORITY",
    "INTERGRAX_ALLOWED_READ_ROOTS",
    "LOCAL_WORKSPACE_DATA_HOME",
    "LKW_DATA_HOME",
)


@pytest.fixture(autouse=True)
def _clear_watcher_env() -> Generator[None, None, None]:
    saved = {key: os.environ[key] for key in _WATCHER_ENV_KEYS if key in os.environ}
    for key in _WATCHER_ENV_KEYS:
        os.environ.pop(key, None)
    yield
    for key in _WATCHER_ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in saved.items():
        os.environ[key] = value


def _enabled_settings(**overrides: object) -> LocalWorkspaceBackendSettings:
    payload: dict[str, object] = {
        "file_watcher_enabled": True,
        "file_watcher_tenant_id": "tenant-a",
        "file_watcher_workspace_id": "workspace-a",
        "file_watcher_collection_id": "collection-a",
        "allowed_read_roots": frozenset({str(Path.cwd().resolve())}),
        "data_home": "build/local_workspace",
        "file_watcher_poll_interval_seconds": 1.0,
        "file_watcher_debounce_seconds": 1.0,
        "file_watcher_max_batch_wait_seconds": 10.0,
        "file_watcher_priority": "normal",
    }
    payload.update(overrides)
    return LocalWorkspaceBackendSettings(**payload)  # type: ignore[arg-type]


def test_file_watcher_disabled_by_default() -> None:
    settings = _settings_from_env()
    assert settings.file_watcher_enabled is False


def test_file_watcher_default_timing_and_priority() -> None:
    settings = _settings_from_env()
    assert settings.file_watcher_poll_interval_seconds == 1.0
    assert settings.file_watcher_debounce_seconds == 1.0
    assert settings.file_watcher_max_batch_wait_seconds == 10.0
    assert settings.file_watcher_priority == "normal"
    assert settings.file_watcher_tenant_id == ""
    assert settings.file_watcher_workspace_id == ""
    assert settings.file_watcher_collection_id == ""


def test_file_watcher_env_values_parse() -> None:
    root = str(Path.cwd().resolve())
    os.environ["LOCAL_WORKSPACE_FILE_WATCHER_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_FILE_WATCHER_TENANT_ID"] = "t1"
    os.environ["LOCAL_WORKSPACE_FILE_WATCHER_WORKSPACE_ID"] = "w1"
    os.environ["LOCAL_WORKSPACE_FILE_WATCHER_COLLECTION_ID"] = "c1"
    os.environ["LOCAL_WORKSPACE_FILE_WATCHER_POLL_INTERVAL_SECONDS"] = "2.5"
    os.environ["LOCAL_WORKSPACE_FILE_WATCHER_DEBOUNCE_SECONDS"] = "3.5"
    os.environ["LOCAL_WORKSPACE_FILE_WATCHER_MAX_BATCH_WAIT_SECONDS"] = "12.0"
    os.environ["LOCAL_WORKSPACE_FILE_WATCHER_PRIORITY"] = "high"
    os.environ["INTERGRAX_ALLOWED_READ_ROOTS"] = root
    os.environ["LOCAL_WORKSPACE_DATA_HOME"] = "custom/lkw"

    settings = _settings_from_env()

    assert settings.file_watcher_enabled is True
    assert settings.file_watcher_tenant_id == "t1"
    assert settings.file_watcher_workspace_id == "w1"
    assert settings.file_watcher_collection_id == "c1"
    assert settings.file_watcher_poll_interval_seconds == 2.5
    assert settings.file_watcher_debounce_seconds == 3.5
    assert settings.file_watcher_max_batch_wait_seconds == 12.0
    assert settings.file_watcher_priority == "high"
    assert settings.allowed_read_roots == frozenset({root})
    assert settings.data_home == "custom/lkw"


def test_watcher_roots_come_from_intergrax_allowed_read_roots() -> None:
    root_a = str((Path.cwd() / "a").resolve())
    root_b = str((Path.cwd() / "b").resolve())
    os.environ["INTERGRAX_ALLOWED_READ_ROOTS"] = f"{root_a},{root_b}"

    settings = _settings_from_env()

    assert settings.allowed_read_roots == frozenset({root_a, root_b})


def test_data_home_resolution_inputs_unchanged() -> None:
    os.environ["LOCAL_WORKSPACE_DATA_HOME"] = "primary/home"
    os.environ["LKW_DATA_HOME"] = "legacy/home"
    os.environ["LOCAL_WORKSPACE_FILE_WATCHER_ENABLED"] = "true"

    settings = _settings_from_env()

    assert settings.data_home == "primary/home"
    assert settings.enable_rag is True


def test_watcher_fields_do_not_affect_unrelated_host_settings() -> None:
    os.environ["LOCAL_WORKSPACE_FILE_WATCHER_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_FILE_WATCHER_TENANT_ID"] = "t1"

    settings = _settings_from_env()

    assert settings.backend_port == 8020
    assert settings.default_agent_id == "local_search"
    assert settings.include_interaction_routes is False


def test_config_builder_rejects_disabled() -> None:
    settings = _enabled_settings(file_watcher_enabled=False)
    with pytest.raises(
        FileWatcherSidecarConfigurationError, match="file_watcher_disabled"
    ):
        build_file_watcher_sidecar_config(settings)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("file_watcher_tenant_id", ""),
        ("file_watcher_tenant_id", "   "),
        ("file_watcher_workspace_id", ""),
        ("file_watcher_collection_id", ""),
    ],
)
def test_config_builder_rejects_blank_identity(field_name: str, value: str) -> None:
    settings = _enabled_settings(**{field_name: value})
    with pytest.raises(
        FileWatcherSidecarConfigurationError,
        match="file_watcher_identity_not_configured",
    ):
        build_file_watcher_sidecar_config(settings)


def test_config_builder_rejects_empty_roots() -> None:
    settings = _enabled_settings(allowed_read_roots=frozenset())
    with pytest.raises(
        FileWatcherSidecarConfigurationError,
        match="file_watcher_roots_not_configured",
    ):
        build_file_watcher_sidecar_config(settings)


def test_config_builder_rejects_blank_root() -> None:
    settings = _enabled_settings(allowed_read_roots=frozenset({"   "}))
    with pytest.raises(
        FileWatcherSidecarConfigurationError,
        match="file_watcher_roots_not_configured",
    ):
        build_file_watcher_sidecar_config(settings)


def test_config_builder_canonicalizes_relative_roots() -> None:
    settings = _enabled_settings(allowed_read_roots=frozenset({"rel-root"}))
    config = build_file_watcher_sidecar_config(settings)
    expected = str(Path("rel-root").expanduser().resolve(strict=False))
    assert config.runtime_config.allowed_roots == frozenset({expected})
    assert Path(next(iter(config.runtime_config.allowed_roots))).is_absolute()


def test_config_builder_rejects_invalid_poll_interval() -> None:
    for value in (0.0, -1.0, math.nan, math.inf):
        settings = _enabled_settings(file_watcher_poll_interval_seconds=value)
        with pytest.raises(
            FileWatcherSidecarConfigurationError,
            match="file_watcher_poll_interval_invalid",
        ):
            build_file_watcher_sidecar_config(settings)


def test_config_builder_rejects_invalid_debounce() -> None:
    for value in (0.0, -1.0, math.nan):
        settings = _enabled_settings(file_watcher_debounce_seconds=value)
        with pytest.raises(
            FileWatcherSidecarConfigurationError,
            match="file_watcher_debounce_invalid",
        ):
            build_file_watcher_sidecar_config(settings)


def test_config_builder_rejects_max_wait_below_debounce() -> None:
    settings = _enabled_settings(
        file_watcher_debounce_seconds=5.0,
        file_watcher_max_batch_wait_seconds=4.0,
    )
    with pytest.raises(
        FileWatcherSidecarConfigurationError,
        match="file_watcher_max_batch_wait_invalid",
    ):
        build_file_watcher_sidecar_config(settings)


def test_config_builder_rejects_invalid_priority() -> None:
    settings = _enabled_settings(file_watcher_priority="urgent")
    with pytest.raises(
        FileWatcherSidecarConfigurationError,
        match="file_watcher_priority_invalid",
    ):
        build_file_watcher_sidecar_config(settings)


@pytest.mark.parametrize("priority", ["low", "normal", "high"])
def test_config_builder_accepts_priorities(priority: str) -> None:
    settings = _enabled_settings(file_watcher_priority=priority)
    config = build_file_watcher_sidecar_config(settings)
    assert config.runtime_config.priority == priority


def test_relative_data_home_resolves_against_working_directory() -> None:
    work = Path.cwd().resolve()
    settings = _enabled_settings(data_home="build/local_workspace")
    config = build_file_watcher_sidecar_config(settings, working_directory=work)
    expected = (
        work / "build" / "local_workspace" / "data" / "file_watcher" / "checkpoint.json"
    )
    assert config.checkpoint_path == expected.resolve(strict=False)
    assert config.checkpoint_path.is_absolute()


def test_absolute_data_home_remains_absolute() -> None:
    absolute_home = (Path.cwd() / "abs-home-for-watcher-test").resolve()
    settings = _enabled_settings(data_home=str(absolute_home))
    config = build_file_watcher_sidecar_config(settings, working_directory=Path.cwd())
    assert (
        config.checkpoint_path
        == absolute_home / "data" / "file_watcher" / "checkpoint.json"
    )
    assert config.checkpoint_path.is_absolute()
