# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit operator wiring for OTLP observability export (OBS-EXPORT-4C)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Mapping

from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy
from intergrax.runtime.observability.export_wiring import make_observability_export_runtime_plugin
from intergrax.runtime.observability.otlp_exporter import (
    OtlpObservabilityExporter,
    OtlpObservabilityExporterConfig,
    OtlpTransport,
)
from intergrax.runtime.observability.otlp_http_transport import OtlpHttpTransport
from intergrax.runtime.plugins.contract import RuntimePlugin

_BACKEND_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

ObservabilityExportBackendBuilder = Callable[..., object]


class ObservabilityExportOperatorConfigError(ValueError):
    """Invalid operator export configuration."""


class ObservabilityExportBackendRegistryError(ValueError):
    """Invalid observability export backend registry operation."""


def parse_observability_export_backend_id(raw: str) -> str:
    """Parse and normalize an open observability export backend identifier."""
    normalized = raw.strip().lower()
    if not normalized or not _BACKEND_ID_PATTERN.match(normalized):
        raise ObservabilityExportOperatorConfigError(
            f"invalid observability export backend id: {raw!r}"
        )
    return normalized


class ObservabilityExportBackendRegistry:
    """Open registry of observability export backend builders keyed by backend_id."""

    def __init__(self) -> None:
        self._builders: dict[str, ObservabilityExportBackendBuilder] = {}

    def register(self, backend_id: str, builder: ObservabilityExportBackendBuilder) -> None:
        normalized = parse_observability_export_backend_id(backend_id)
        if normalized in self._builders:
            raise ObservabilityExportBackendRegistryError(
                f"observability export backend builder already registered for {normalized!r}"
            )
        self._builders[normalized] = builder

    def get(self, backend_id: str) -> ObservabilityExportBackendBuilder:
        normalized = parse_observability_export_backend_id(backend_id)
        try:
            return self._builders[normalized]
        except KeyError as exc:
            raise ObservabilityExportBackendRegistryError(
                f"no observability export backend builder registered for {normalized!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class OtlpExportOperatorConfig:
    endpoint: str
    service_name: str
    service_version: str | None = None
    environment: str | None = None
    timeout_seconds: float | None = None
    headers: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ObservabilityExportOperatorConfig:
    enabled: bool = False
    export_content: bool = False
    backend_id: str = "otlp"
    otlp: OtlpExportOperatorConfig | None = None


def _require_enabled_otlp_config(config: ObservabilityExportOperatorConfig) -> OtlpExportOperatorConfig:
    if not config.enabled:
        raise ObservabilityExportOperatorConfigError("observability export is disabled")
    if config.backend_id != "otlp":
        raise ObservabilityExportOperatorConfigError(
            f"otlp export configuration requires backend_id 'otlp', got {config.backend_id!r}"
        )
    if config.otlp is None:
        raise ObservabilityExportOperatorConfigError("otlp export configuration is required")
    return config.otlp


def _build_otlp_exporter_config(otlp: OtlpExportOperatorConfig) -> OtlpObservabilityExporterConfig:
    return OtlpObservabilityExporterConfig(
        endpoint=otlp.endpoint,
        service_name=otlp.service_name,
        service_version=otlp.service_version or "",
        environment=otlp.environment or "",
        timeout_seconds=otlp.timeout_seconds if otlp.timeout_seconds is not None else 30.0,
        headers=dict(otlp.headers) if otlp.headers is not None else {},
    )


def build_otlp_observability_integration(
    config: ObservabilityExportOperatorConfig,
    *,
    transport: OtlpTransport | None = None,
):
    """Construct an OTLP observability vendor integration from explicit operator configuration."""
    from intergrax.runtime.integrations.observability_otlp import OtlpObservabilityIntegration

    otlp = _require_enabled_otlp_config(config)
    exporter_config = _build_otlp_exporter_config(otlp)
    active_transport = transport or OtlpHttpTransport()
    exporter = OtlpObservabilityExporter(exporter_config, active_transport)
    return OtlpObservabilityIntegration.from_exporter(exporter, enabled=config.enabled)


def _build_otlp_observability_integration_from_registry(
    config: ObservabilityExportOperatorConfig,
    *,
    transport: OtlpTransport | None = None,
):
    return build_otlp_observability_integration(config, transport=transport)


DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY = ObservabilityExportBackendRegistry()
DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY.register("otlp", _build_otlp_observability_integration_from_registry)


def build_observability_export_integration(
    config: ObservabilityExportOperatorConfig,
    *,
    transport: OtlpTransport | None = None,
    registry: ObservabilityExportBackendRegistry | None = None,
):
    """Construct an observability vendor integration via the open backend builder registry."""
    active_registry = registry or DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY
    builder = active_registry.get(config.backend_id)
    return builder(config, transport=transport)


def build_otlp_observability_exporter(
    config: ObservabilityExportOperatorConfig,
    *,
    transport: OtlpTransport | None = None,
) -> OtlpObservabilityExporter:
    """Construct an OTLP exporter from explicit operator configuration."""
    return build_otlp_observability_integration(config, transport=transport).exporter


def build_observability_export_runtime_plugin(
    config: ObservabilityExportOperatorConfig,
    *,
    transport: OtlpTransport | None = None,
    registry: ObservabilityExportBackendRegistry | None = None,
) -> RuntimePlugin | None:
    """Construct a runtime export plugin from explicit operator configuration."""
    if not config.enabled:
        return None

    integration = build_observability_export_integration(
        config,
        transport=transport,
        registry=registry,
    )
    policy = ObservabilityExportPolicy(enabled=True, export_content=False)
    return make_observability_export_runtime_plugin(exporter=integration, policy=policy)


def build_otlp_observability_export_runtime_plugin(
    config: ObservabilityExportOperatorConfig,
    *,
    transport: OtlpTransport | None = None,
) -> RuntimePlugin | None:
    """Construct an OTLP runtime export plugin from explicit operator configuration."""
    if not config.enabled:
        return None

    integration = build_otlp_observability_integration(config, transport=transport)
    policy = ObservabilityExportPolicy(enabled=True, export_content=False)
    return make_observability_export_runtime_plugin(exporter=integration, policy=policy)
