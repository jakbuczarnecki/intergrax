# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 Nexus platform bootstrap (plugins, shutdown hooks)."""

from __future__ import annotations

from typing import Optional

from intergrax.applications._shared.plugin_bootstrap import (
    PluginBootstrapResult,
    attach_plugin_shutdown,
    bootstrap_application_plugins,
)
from intergrax.runtime.governance.contracts.metrics_store import ExecutionMetricsStore
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader
from intergrax.llm_adapters.tracking.observability_bridge import register_llm_observability_plugin
from intergrax.runtime.plugins.default_plugins import default_lab_plugins


def bootstrap_nexus_platform(
    nexus_loop: NexusLoop,
    *,
    trace_store: Optional[RunTraceReader] = None,
    metrics_store: Optional[ExecutionMetricsStore] = None,
) -> PluginBootstrapResult:
    """Register default runtime plugins on a composed NexusLoop."""
    reader = trace_store
    if reader is None and hasattr(nexus_loop, "trace_emitter"):
        emitter = nexus_loop.trace_emitter
        if emitter is not None and hasattr(emitter, "trace_store"):
            reader = emitter.trace_store  # type: ignore[attr-defined]
    plugins = default_lab_plugins(trace_store=reader, metrics_store=metrics_store)
    register_llm_observability_plugin(plugins)
    return bootstrap_application_plugins(plugins, nexus_loop=nexus_loop)
