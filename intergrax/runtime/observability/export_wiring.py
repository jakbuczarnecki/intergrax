# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Minimal runtime lifecycle wiring for normalized observability export (OBS-EXPORT-2)."""

from __future__ import annotations

from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.observability.export_boundary import (
    NoOpObservabilityExporter,
    ObservabilityExporter,
    envelope_from_runtime_event,
)
from intergrax.runtime.observability.export_policy import (
    ObservabilityExportPolicy,
    try_export_observability_envelope,
)
from intergrax.runtime.plugins.contract import PolicyEngineLike, RuntimeEventBusLike, RuntimePlugin


def make_observability_export_runtime_plugin(
    *,
    exporter: ObservabilityExporter | None = None,
    policy: ObservabilityExportPolicy | None = None,
) -> RuntimePlugin:
    """Runtime plugin — optional normalized export after canonical bus recording."""

    active_exporter = exporter or NoOpObservabilityExporter()
    active_policy = policy or ObservabilityExportPolicy()

    def _register(
        event_bus: RuntimeEventBusLike,
        _hook_registry: HookRegistry,
        _policy_engine: PolicyEngineLike,
    ) -> None:
        if not active_policy.enabled:
            return

        async def _export_runtime_event(event: RuntimeEvent) -> None:
            envelope = envelope_from_runtime_event(event)
            await try_export_observability_envelope(
                envelope,
                exporter=active_exporter,
                policy=active_policy,
            )

        event_bus.subscribe(
            _export_runtime_event,
            subscription_id="plugin.observability_export",
        )

    return RuntimePlugin(
        plugin_id="runtime.observability_export",
        version="1.0.0",
        register=_register,
    )
