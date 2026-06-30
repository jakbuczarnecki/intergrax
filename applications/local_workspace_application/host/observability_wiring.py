# © Artur Czarnecki. All rights reserved.

"""Explicit platform observability export wiring for local_workspace_application."""

from __future__ import annotations

from intergrax.runtime.observability.operator_wiring import (
    ObservabilityExportOperatorConfig,
    build_observability_export_runtime_plugin,
)
from intergrax.runtime.observability.otlp_exporter import OtlpTransport
from intergrax.runtime.plugins.contract import RuntimePlugin


def build_local_workspace_observability_plugins(
    observability_export: ObservabilityExportOperatorConfig | None,
    *,
    transport: OtlpTransport | None = None,
) -> tuple[RuntimePlugin, ...]:
    """Compose LKW runtime observability export plugins from explicit platform operator config."""
    if observability_export is None or not observability_export.enabled:
        return ()
    plugin = build_observability_export_runtime_plugin(
        observability_export,
        transport=transport,
    )
    return (plugin,) if plugin is not None else ()
