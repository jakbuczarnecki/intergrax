# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sentry observability export operator wiring (LKW-OBS-SENTRY-0)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.runtime.observability.operator_wiring import (
    ObservabilityExportOperatorConfig,
    ObservabilityExportOperatorConfigError,
    SentryExportOperatorConfig,
)

if TYPE_CHECKING:
    from intergrax.integrations.providers.observability_backend.sentry.client import (
        SentryCaptureClient,
    )
    from intergrax.integrations.providers.observability_backend.sentry.integration import (
        SentryObservabilityIntegration,
        SentryObservabilityTransport,
    )


def _require_enabled_sentry_config(
    config: ObservabilityExportOperatorConfig,
) -> SentryExportOperatorConfig:
    if not config.enabled:
        raise ObservabilityExportOperatorConfigError("observability export is disabled")
    if config.backend_id != "sentry":
        raise ObservabilityExportOperatorConfigError(
            f"sentry export configuration requires backend_id 'sentry', got {config.backend_id!r}"
        )
    if config.sentry is None:
        raise ObservabilityExportOperatorConfigError("sentry export configuration is required")
    sentry = config.sentry
    if not sentry.dsn.strip():
        raise ObservabilityExportOperatorConfigError("sentry dsn is required")
    if (
        sentry.shutdown_timeout_seconds is not None
        and sentry.shutdown_timeout_seconds < 0
    ):
        raise ObservabilityExportOperatorConfigError(
            "sentry shutdown_timeout_seconds must be >= 0"
        )
    return sentry


def build_sentry_observability_integration(
    config: ObservabilityExportOperatorConfig,
    *,
    transport: SentryObservabilityTransport | None = None,
    client: SentryCaptureClient | None = None,
) -> SentryObservabilityIntegration:
    """Construct a Sentry observability vendor integration from operator config."""
    from intergrax.integrations.providers.observability_backend.sentry.bundle import (
        create_sentry_observability_integration,
        create_sentry_observability_transport,
    )

    sentry = _require_enabled_sentry_config(config)
    config_overrides: dict[str, object] = {"dsn": sentry.dsn}
    if sentry.environment is not None:
        config_overrides["environment"] = sentry.environment
    if sentry.release is not None:
        config_overrides["release"] = sentry.release
    if sentry.server_name is not None:
        config_overrides["server_name"] = sentry.server_name
    if sentry.shutdown_timeout_seconds is not None:
        config_overrides["shutdown_timeout_seconds"] = sentry.shutdown_timeout_seconds
    config_overrides["debug"] = sentry.debug

    active_transport = transport or create_sentry_observability_transport(
        client=client,
        flush_after_capture=sentry.flush_after_capture,
        **config_overrides,
    )
    return create_sentry_observability_integration(
        transport=active_transport,
        enabled=config.enabled,
    )


def _build_default_sentry_observability_integration(
    config: ObservabilityExportOperatorConfig,
) -> object:
    return build_sentry_observability_integration(config)
