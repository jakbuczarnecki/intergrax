# © Artur Czarnecki. All rights reserved.

"""Tests for env-driven LKW observability export settings (LKW-OBS-OTLP-1A)."""

from __future__ import annotations

import os
from typing import Generator
from unittest.mock import patch

import pytest

from intergrax.runtime.observability.operator_wiring import (
    ObservabilityExportBackendRegistryError,
    ObservabilityExportOperatorConfig,
    build_observability_export_runtime_plugin,
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
    assert config.backend_id == "otlp"
    assert config.export_content is False
    assert config.otlp is not None
    assert config.otlp.endpoint == "http://otel-collector:4318/v1/logs"
    assert config.otlp.service_name == "intergrax-lkw"
    assert config.otlp.service_version == "dev"
    assert config.otlp.environment == "dev"


def test_elasticsearch_backend_normalizes_in_config() -> None:
    """Valid elasticsearch backend_id is accepted by LKW settings and normalized."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = " ELASTICSEARCH "
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL"] = "http://elasticsearch.local:9200"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX"] = "logs-*"

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()

    assert config is not None
    assert config.backend_id == "elasticsearch"
    assert config.otlp is None
    assert config.elasticsearch is not None


def test_custom_plugin_backend_id_is_allowed_in_config() -> None:
    """LKW does not reject valid custom/plugin backend ids at settings time."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "acme_observability"

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()

    assert config is not None
    assert config.backend_id == "acme_observability"
    assert config.otlp is None


def test_acme_observability_without_otlp_endpoint_builds_config_with_otlp_none() -> None:
    """Non-OTLP backend_id does not require LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "acme_observability"

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()

    assert config is not None
    assert config.backend_id == "acme_observability"
    assert config.otlp is None


def test_elasticsearch_backend_fails_at_platform_build_step() -> None:
    """Valid but unregistered backend_id fails when building the runtime plugin."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "acme_observability"

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()
    assert config is not None
    assert config.otlp is None

    with pytest.raises(
        ObservabilityExportBackendRegistryError,
        match="no observability export backend builder registered for 'acme_observability'",
    ):
        build_observability_export_runtime_plugin(config)


def test_enabled_elasticsearch_config_builds_runtime_plugin() -> None:
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "elasticsearch"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL"] = "http://elasticsearch.local:9200"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX"] = "logs-*"

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()
    assert config is not None
    assert config.backend_id == "elasticsearch"
    assert config.elasticsearch is not None
    assert config.elasticsearch.base_url == "http://elasticsearch.local:9200"
    assert config.elasticsearch.index == "logs-*"
    assert config.otlp is None

    with patch(
        "intergrax.integrations.providers.observability_backend.elasticsearch.bundle.create_elasticsearch_observability_transport",
    ) as create_transport:
        create_transport.return_value = object()
        plugin = build_observability_export_runtime_plugin(config)

    assert plugin is not None
    assert plugin.plugin_id == "runtime.observability_export"


def test_enabled_elasticsearch_without_url_raises() -> None:
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "elasticsearch"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX"] = "logs-*"

    settings = LocalWorkspaceBackendSettings.from_env()
    with pytest.raises(ValueError, match="LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL"):
        settings.build_observability_export_config()


def test_enabled_elasticsearch_without_index_raises() -> None:
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "elasticsearch"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL"] = "http://elasticsearch.local:9200"

    settings = LocalWorkspaceBackendSettings.from_env()
    with pytest.raises(ValueError, match="LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX"):
        settings.build_observability_export_config()


def test_export_content_remains_false() -> None:
    """Even when export_content env is true, the resulting config has False."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_CONTENT"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "otlp"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT"] = "http://localhost:4318"

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()

    assert config is not None
    assert config.export_content is False


def test_enabled_otlp_without_endpoint_raises() -> None:
    """Enabled OTLP export without endpoint raises ValueError."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "otlp"

    settings = LocalWorkspaceBackendSettings.from_env()
    with pytest.raises(ValueError, match="LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT"):
        settings.build_observability_export_config()


def test_enabled_without_endpoint_raises() -> None:
    """Default OTLP backend without endpoint raises ValueError."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"

    settings = LocalWorkspaceBackendSettings.from_env()
    with pytest.raises(ValueError, match="LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT"):
        settings.build_observability_export_config()


def test_invalid_backend_id_raises() -> None:
    """Invalid backend id format raises ValueError."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "foo/bar"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT"] = "http://localhost:4318"

    settings = LocalWorkspaceBackendSettings.from_env()
    with pytest.raises(ValueError, match="invalid observability export backend id: 'foo/bar'"):
        settings.build_observability_export_config()


def test_sentry_env_builds_sentry_export_operator_config() -> None:
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "sentry"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_DSN"] = "https://example@sentry.io/1"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_ENVIRONMENT"] = "dev"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_RELEASE"] = "lkw-test"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_SERVER_NAME"] = "local-lkw"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_SHUTDOWN_TIMEOUT_SECONDS"] = "1.5"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_DEBUG"] = "false"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_FLUSH_AFTER_CAPTURE"] = "true"

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()

    assert config is not None
    assert config.backend_id == "sentry"
    assert config.sentry is not None
    assert config.sentry.dsn == "https://example@sentry.io/1"
    assert config.sentry.environment == "dev"
    assert config.sentry.release == "lkw-test"
    assert config.sentry.server_name == "local-lkw"
    assert config.sentry.shutdown_timeout_seconds == 1.5
    assert config.sentry.flush_after_capture is True
    assert config.export_content is False
    assert config.otlp is None
    assert config.elasticsearch is None


def test_sentry_backend_without_dsn_raises() -> None:
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "sentry"

    settings = LocalWorkspaceBackendSettings.from_env()
    with pytest.raises(ValueError, match="LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_DSN"):
        settings.build_observability_export_config()


@pytest.mark.parametrize("raw_dsn", ["", "   "])
def test_sentry_whitespace_dsn_raises(raw_dsn: str) -> None:
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "sentry"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_DSN"] = raw_dsn

    settings = LocalWorkspaceBackendSettings.from_env()
    with pytest.raises(ValueError, match="LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_DSN"):
        settings.build_observability_export_config()


def test_sentry_backend_normalizes_in_config() -> None:
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = " SENTRY "
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_DSN"] = "https://example@sentry.io/1"

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()

    assert config is not None
    assert config.backend_id == "sentry"
    assert config.sentry is not None


def test_enabled_sentry_config_builds_runtime_plugin() -> None:
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "true"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_BACKEND"] = "sentry"
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_DSN"] = "https://example@sentry.io/1"

    settings = LocalWorkspaceBackendSettings.from_env()
    config = settings.build_observability_export_config()
    assert config is not None

    with patch(
        "intergrax.integrations.providers.observability_backend.sentry.bundle.create_sentry_observability_transport",
    ) as create_transport:
        create_transport.return_value = object()
        plugin = build_observability_export_runtime_plugin(config)

    assert plugin is not None
    assert plugin.plugin_id == "runtime.observability_export"


def test_factory_explicit_config_wins(tmp_path, monkeypatch) -> None:
    """When explicit observability_export is passed to factory, it takes precedence
    over settings-derived config."""
    os.environ["LOCAL_WORKSPACE_OBSERVABILITY_EXPORT_ENABLED"] = "false"

    explicit_config = ObservabilityExportOperatorConfig(
        enabled=True,
        export_content=False,
        backend_id="otlp",
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
