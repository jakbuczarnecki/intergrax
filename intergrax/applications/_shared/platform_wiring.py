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
from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationPluginBootstrapTarget,
)
from intergrax.contracts.run_trace_store import RunTraceReader
from intergrax.runtime.task.task_trace import PersistingTaskTraceEmitter
from intergrax.llm_adapters.tracking.observability_bridge import register_llm_observability_plugin
from intergrax.rag.tracking.observability_bridge import register_rag_observability_plugin
from intergrax.runtime.observability.export_bridge import register_journal_export_plugin
from intergrax.runtime.plugins.default_plugins import default_lab_plugins


def bootstrap_nexus_platform(
    orchestration_host: HostOrchestrationPluginBootstrapTarget,
    *,
    trace_store: Optional[RunTraceReader] = None,
    metrics_store: Optional[ExecutionMetricsStore] = None,
) -> PluginBootstrapResult:
    """Register default runtime plugins on a composed orchestration host."""
    reader = trace_store or orchestration_host.trace_store
    if reader is None:
        emitter = orchestration_host.trace_emitter
        if isinstance(emitter, PersistingTaskTraceEmitter):
            reader = emitter.trace_store
    plugins = default_lab_plugins(trace_store=reader, metrics_store=metrics_store)
    register_llm_observability_plugin(plugins)
    register_rag_observability_plugin(plugins)
    register_journal_export_plugin(
        plugins,
        trace_store=reader,
        runtime_event_store=orchestration_host.runtime_event_store,
    )
    return bootstrap_application_plugins(plugins, orchestration_host=orchestration_host)
