# © Artur Czarnecki. All rights reserved.

"""Explicit platform observability export wiring for local_workspace_application."""

from __future__ import annotations

from intergrax.runtime.observability.export_boundary import (
    NoOpObservabilityExporter,
    ObservabilityExporter,
)
from intergrax.runtime.observability.operator_wiring import (
    ObservabilityExportBackendRegistry,
    ObservabilityExportOperatorConfig,
    build_observability_export_integration,
    build_observability_export_runtime_plugin,
)
from intergrax.runtime.plugins.contract import RuntimePlugin
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings


def resolve_local_workspace_observability_exporter(
    settings: LocalWorkspaceBackendSettings,
    *,
    registry: ObservabilityExportBackendRegistry | None = None,
) -> ObservabilityExporter:
    """Resolve explicit LKW production observability exporter from platform operator config."""
    config = settings.build_observability_export_config()
    if config is None or not config.enabled:
        return NoOpObservabilityExporter()
    integration = build_observability_export_integration(config, registry=registry)
    if not isinstance(integration, ObservabilityExporter):
        raise TypeError("observability export integration must implement ObservabilityExporter")
    return integration


def build_local_workspace_observability_plugins(
    observability_export: ObservabilityExportOperatorConfig | None,
    *,
    registry: ObservabilityExportBackendRegistry | None = None,
) -> tuple[RuntimePlugin, ...]:
    """Compose LKW runtime observability export plugins from explicit platform operator config."""
    if observability_export is None or not observability_export.enabled:
        return ()
    plugin = build_observability_export_runtime_plugin(
        observability_export,
        registry=registry,
    )
    return (plugin,) if plugin is not None else ()
