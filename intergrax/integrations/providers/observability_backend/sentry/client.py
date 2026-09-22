# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sentry SDK capture client — the only module that may import sentry_sdk (OBS-SENTRY-1)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, no_type_check, runtime_checkable

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.sentry.config import SentryIntegrationConfig


@runtime_checkable
class SentrySdkFacade(Protocol):
    """Minimal lazy-imported sentry_sdk module surface used by this provider."""

    def capture_event(self, event: dict[str, object]) -> object:
        """Capture one SDK event payload."""
        ...

    def flush(self, timeout: float | None = None) -> None:
        """Flush pending SDK events."""
        ...


@runtime_checkable
class SentryCaptureClient(Protocol):
    """Provider-owned capture facade used by the observability transport."""

    def capture_event(self, event: Mapping[str, object]) -> str | None:
        """Capture one policy-safe Sentry event and return the event id when available."""

    def flush(self, timeout: float | None = None) -> None:
        """Flush pending Sentry events."""


class _SentrySdkModuleAdapter:
    """Provider-owned binding from optional sentry_sdk callables to SentrySdkFacade."""

    def __init__(
        self,
        *,
        capture_event: Callable[[dict[str, object]], object],
        flush: Callable[[float | None], None],
    ) -> None:
        self._capture_event = capture_event
        self._flush = flush

    def capture_event(self, event: dict[str, object]) -> object:
        return self._capture_event(event)

    def flush(self, timeout: float | None = None) -> None:
        self._flush(timeout)


class SentrySdkCaptureClient:
    """Lazy sentry_sdk-backed capture client — SDK import happens only here."""

    def __init__(self, *, _sdk: SentrySdkFacade) -> None:
        self._sdk = _sdk

    @classmethod
    def from_config(cls, config: SentryIntegrationConfig) -> SentrySdkCaptureClient:
        if not config.dsn:
            raise IntegrationConfigurationError(
                "Sentry SDK client requires a DSN in provider configuration",
            )
        try:
            import sentry_sdk as sentry_sdk_module
        except ImportError as exc:
            raise IntegrationConfigurationError(
                "Sentry SDK reporting requires sentry-sdk. Install with: uv pip install sentry-sdk",
            ) from exc

        capture_binding: object = sentry_sdk_module.capture_event
        flush_binding: object = sentry_sdk_module.flush

        @no_type_check
        def _capture_event(event: dict[str, object]) -> object:
            return capture_binding(dict(event))

        @no_type_check
        def _flush_sdk(timeout: float | None = None) -> None:
            flush_binding(timeout=timeout)

        sdk = _SentrySdkModuleAdapter(
            capture_event=_capture_event,
            flush=_flush_sdk,
        )
        sentry_sdk_module.init(
            dsn=config.dsn,
            environment=config.environment or None,
            release=config.release or None,
            server_name=config.server_name or None,
            send_default_pii=False,
            attach_stacktrace=False,
            debug=config.debug,
        )
        return cls(_sdk=sdk)

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
