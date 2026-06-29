# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit operator wiring for OTLP observability export (OBS-EXPORT-4C)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
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


class ObservabilityExportBackend(StrEnum):
    OTLP = "otlp"


class ObservabilityExportOperatorConfigError(ValueError):
    """Invalid operator export configuration."""


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
    backend: ObservabilityExportBackend = ObservabilityExportBackend.OTLP
    otlp: OtlpExportOperatorConfig | None = None


def _require_enabled_otlp_config(config: ObservabilityExportOperatorConfig) -> OtlpExportOperatorConfig:
    if not config.enabled:
        raise ObservabilityExportOperatorConfigError("observability export is disabled")
    if config.backend is not ObservabilityExportBackend.OTLP:
        raise ObservabilityExportOperatorConfigError(
            f"unsupported observability export backend: {config.backend.value}"
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


def build_otlp_observability_exporter(
    config: ObservabilityExportOperatorConfig,
    *,
    transport: OtlpTransport | None = None,
) -> OtlpObservabilityExporter:
    """Construct an OTLP exporter from explicit operator configuration."""
    return build_otlp_observability_integration(config, transport=transport).exporter


def build_otlp_observability_export_runtime_plugin(
    config: ObservabilityExportOperatorConfig,
    *,
    transport: OtlpTransport | None = None,
) -> RuntimePlugin | None:
    """Construct a runtime export plugin from explicit operator configuration."""
    if not config.enabled:
        return None

    integration = build_otlp_observability_integration(config, transport=transport)
    policy = ObservabilityExportPolicy(enabled=True, export_content=False)
    return make_observability_export_runtime_plugin(exporter=integration, policy=policy)
