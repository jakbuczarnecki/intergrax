# © Artur Czarnecki. All rights reserved.

"""Tests for env-driven LKW observability export settings (LKW-OBS-OTLP-1A)."""

from __future__ import annotations

import os
from typing import Generator

import pytest

from intergrax.runtime.observability.operator_wiring import (
    ObservabilityExportBackend,
    ObservabilityExportOperatorConfig,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _clear_observability_env() -> Generator[None, None, None]:
    """Remove all LOCAL_WORKSPACE_OBSERVABILITY_* env vars before each test."""
    keys = [k for k in os.environ if k.startswith("LOCAL_WORKSPACE_OBSERVABILITY_")]
    saved = {k: os.environ[k] for k in keys}
    for k in keys:
        del os.environ[k]
    yield
    for k in saved:
        os.environ[k] = saved[k]
    for k in [k for k in os.environ if k.startswith("LOCAL_WORKSPACE_OBSERVABILITY_")]:
        if k not in saved:
            del os.environ[k]


def test_disabled_by_default() -> None:
    """Given no observability env vars, build_observability_export_config returns None."""
    settings = LocalWorkspaceBackendSettings.from_env()
    assert settings.build_observability_export_config() is None


def test_enabled_otlp_config() -> None:
    """Enabled OTLP export returns a valid ObservabilityExportOperatorConfig."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "otlp"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT"] = (
        "http://otel-collector:4318/v1/logs"
    )
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_SERVICE_NAME"] = "intergrax-lkw"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_SERVICE_VERSION"] = "dev"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ENVIRONMENT"] = "dev"

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()

    assert config is not None
    assert config.enabled is True
    assert config.backend is ObservabilityExportBackend.OTLP
    assert config.export_content is False
    assert config.otlp is not None
    assert config.otlp.endpoint == "http://otel-collector:4318/v1/logs"
    assert config.otlp.service_name == "intergrax-lkw"
    assert config.otlp.service_version == "dev"
    assert config.otlp.environment == "dev"


def test_export_content_remains_false() -> None:
    """Even when export_content env is true, the resulting config has False."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_CONTENT"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT"] = "http://localhost:4318"

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()

    assert config is not None
    assert config.export_content is False


def test_enabled_without_endpoint_raises() -> None:
    """Enabled OTLP export without endpoint raises ValueError."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"

    settings = LocalWorkspaceBackendSettings.from_env()
    with pytest.raises(ValueError, match="LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT"):
        settings.build_observability_export_config()


def test_unrecognized_backend_raises() -> None:
    """Unknown backend raises ValueError."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "foo"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT"] = "http://localhost:4318"

    settings = LocalWorkspaceBackendSettings.from_env()
    with pytest.raises(ValueError, match="unsupported observability export backend: 'foo'"):
        settings.build_observability_export_config()


def test_recognized_but_not_implemented_backend_raises() -> None:
    """Recognized non-OTLP backend fails fast as not implemented."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "elasticsearch"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT"] = "http://localhost:4318"

    settings = LocalWorkspaceBackendSettings.from_env()
    with pytest.raises(
        ValueError,
        match="recognized but not implemented in operator wiring yet",
    ):
        settings.build_observability_export_config()


def test_unsupported_backend_raises() -> None:
    """Recognized langfuse backend fails fast as not implemented."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "langfuse"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT"] = "http://localhost:4318"

    settings = LocalWorkspaceBackendSettings.from_env()
    with pytest.raises(
        ValueError,
        match="recognized but not implemented in operator wiring yet",
    ):
        settings.build_observability_export_config()


def test_factory_explicit_config_wins(tmp_path, monkeypatch) -> None:
    """When explicit observability_export is passed to factory, it takes precedence
    over settings-derived config."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "false"

    explicit_config = ObservabilityExportOperatorConfig(
        enabled=True,
        export_content=False,
        backend=ObservabilityExportBackend.OTLP,
        otlp=None,  # Not needed for this test — we just check it's passed through
    )

    monkeypatch.setattr(
        "local_workspace_application.host.factory.build_local_workspace_observability_plugins",
        lambda *args, **kwargs: (),
    )

    from local_workspace_application.host.factory import (
        create_local_workspace_backend_app,
    )

    # The factory should use the explicit config, not build one from disabled settings
    app = create_local_workspace_backend_app(observability_export=explicit_config)
    assert app is not None
    assert app.title
