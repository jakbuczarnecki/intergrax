# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sentry SDK capture client — the only module that may import sentry_sdk (OBS-SENTRY-1)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.sentry.config import SentryIntegrationConfig


@runtime_checkable
class SentryCaptureClient(Protocol):
    """Provider-owned capture facade used by the observability transport."""

    def capture_event(self, event: Mapping[str, object]) -> str | None:
        """Capture one policy-safe Sentry event and return the event id when available."""

    def flush(self, timeout: float | None = None) -> None:
        """Flush pending Sentry events."""


class SentrySdkCaptureClient:
    """Lazy sentry_sdk-backed capture client — SDK import happens only here."""

    def __init__(self, *, _sdk: object) -> None:
        self._sdk = _sdk

    @classmethod
    def from_config(cls, config: SentryIntegrationConfig) -> SentrySdkCaptureClient:
        if not config.dsn:
            raise IntegrationConfigurationError(
                "Sentry SDK client requires a DSN in provider configuration",
            )
        try:
            import sentry_sdk
        except ImportError as exc:
            raise IntegrationConfigurationError(
                "Sentry SDK reporting requires sentry-sdk. Install with: uv pip install sentry-sdk",
            ) from exc

        sentry_sdk.init(
            dsn=config.dsn,
            environment=config.environment or None,
            release=config.release or None,
            server_name=config.server_name or None,
            send_default_pii=False,
            attach_stacktrace=False,
            debug=config.debug,
        )
        return cls(_sdk=sentry_sdk)

    def capture_event(self, event: Mapping[str, object]) -> str | None:
        result = self._sdk.capture_event(dict(event))
        if result is None:
            return None
        return str(result)

    def flush(self, timeout: float | None = None) -> None:
        self._sdk.flush(timeout=timeout)


def open_sentry_sdk_capture_client(config: SentryIntegrationConfig) -> SentrySdkCaptureClient:
    """Open a real Sentry SDK capture client from provider-owned configuration."""
    return SentrySdkCaptureClient.from_config(config)
