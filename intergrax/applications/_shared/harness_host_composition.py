# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Internal harness host orchestration capabilities (NPSC-4.2).

Typed narrow capabilities owned by :class:`HarnessHostRuntime` composition.
Tier-3 authors MUST use ``HarnessHostRuntime.execution`` — not this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.runtime.decision_flow import DecisionFlowGate
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService
from intergrax.runtime.governance.contracts.metrics_store import ExecutionMetricsStore
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.hooks.nexus_lifecycle_hooks import NexusLifecycleHookCoordinator
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationPluginBootstrapTarget,
)
from intergrax.contracts.run_trace_store import RunTraceReader
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedgerFactory
from intergrax.runtime.plugins.bootstrap import PluginBootstrapResult, bootstrap_runtime_plugins
from intergrax.runtime.plugins.contract import RuntimePlugin
from intergrax.runtime.policy.policy_engine import PolicyEngine

if TYPE_CHECKING:
    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime


@dataclass(frozen=True, slots=True)
class HarnessHostPluginRegistrationSurface:
    """Narrow plugin registration inputs owned by harness host composition."""

    event_bus: RuntimeEventBus
    hook_registry: HookRegistry
    policy_engine: PolicyEngine


@dataclass(frozen=True, slots=True)
class HarnessHostInternalComposition:
    """Private orchestration capabilities assembled by harness host composition root."""

    execution_terminal: ExecutionTerminalService
    event_bus: RuntimeEventBus
    decision_flow_gate: DecisionFlowGate[AgentExecutionResult] | None
    middleware_pipeline: MiddlewarePipeline
    lifecycle_hook_coordinator: NexusLifecycleHookCoordinator
    plugin_surface: HarnessHostPluginRegistrationSurface
    runtime_event_persistence: RuntimeEventPersistence | None
    plugin_bootstrap_target: HostOrchestrationPluginBootstrapTarget


def _require_internal_composition(runtime: HarnessHostRuntime) -> HarnessHostInternalComposition:
    return runtime._internal_composition


def resolve_harness_host_execution_terminal(runtime: HarnessHostRuntime) -> ExecutionTerminalService:
    """Resolve canonical execution terminal signaling for host auxiliary wiring."""
    return _require_internal_composition(runtime).execution_terminal


def resolve_harness_host_event_bus(runtime: HarnessHostRuntime) -> RuntimeEventBus:
    """Resolve runtime event bus for host-scoped observability and scaling wiring."""
    return _require_internal_composition(runtime).event_bus


def resolve_harness_host_decision_flow_gate(
    runtime: HarnessHostRuntime,
) -> DecisionFlowGate[AgentExecutionResult] | None:
    """Resolve decision flow gate for ACP session host composition."""
    return _require_internal_composition(runtime).decision_flow_gate


def resolve_harness_host_plugin_bootstrap_target(
    runtime: HarnessHostRuntime,
) -> HostOrchestrationPluginBootstrapTarget:
    """Resolve orchestration host surface for plugin/bootstrap wiring."""
    return _require_internal_composition(runtime).plugin_bootstrap_target


def resolve_harness_host_execution_budget_ledger_factory(
    runtime: HarnessHostRuntime,
) -> ExecutionBudgetLedgerFactory | None:
    return _require_internal_composition(
        runtime,
    ).plugin_bootstrap_target.execution_budget_ledger_factory


def resolve_harness_host_orchestration_topology_submission_port(
    runtime: HarnessHostRuntime,
) -> object:
    """Resolve strict production topology submission port wired on the host runtime."""
    from intergrax.applications._shared.harness_host_orchestration_topology_wiring import (
        resolve_harness_host_orchestration_topology_wiring,
    )

    return resolve_harness_host_orchestration_topology_wiring(runtime).submission_port


def resolve_harness_host_middleware_pipeline(runtime: HarnessHostRuntime) -> MiddlewarePipeline:
    """Resolve middleware pipeline for platform assembly verification."""
    return _require_internal_composition(runtime).middleware_pipeline


def resolve_harness_host_lifecycle_hook_coordinator(
    runtime: HarnessHostRuntime,
) -> NexusLifecycleHookCoordinator:
    """Resolve lifecycle hook coordinator for platform lifecycle verification."""
    return _require_internal_composition(runtime).lifecycle_hook_coordinator


def resolve_harness_host_runtime_event_persistence(
    runtime: HarnessHostRuntime,
) -> RuntimeEventPersistence | None:
    """Resolve runtime event persistence from observability wiring or composition root."""
    store = runtime.observability.runtime_event_store
    if store is not None:
        return store
    return _require_internal_composition(runtime).runtime_event_persistence


def bootstrap_harness_host_platform(
    runtime: HarnessHostRuntime,
    *,
    trace_store: RunTraceReader | None = None,
    metrics_store: ExecutionMetricsStore | None = None,
) -> PluginBootstrapResult:
    """Register default runtime plugins for a composed harness host."""
    from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform

    composition = _require_internal_composition(runtime)
    resolved_trace = trace_store or runtime.observability.trace_store
    return bootstrap_nexus_platform(
        composition.plugin_bootstrap_target,
        trace_store=resolved_trace,
        metrics_store=metrics_store,
    )


def bootstrap_harness_host_application_plugins(
    runtime: HarnessHostRuntime,
    plugins: list[RuntimePlugin],
) -> PluginBootstrapResult:
    """Wire application-specific runtime plugins against harness host composition."""
    surface = _require_internal_composition(runtime).plugin_surface
    return bootstrap_runtime_plugins(
        plugins,
        event_bus=surface.event_bus,
        hook_registry=surface.hook_registry,
        policy_engine=surface.policy_engine,
    )


__all__ = [
    "HarnessHostInternalComposition",
    "HarnessHostPluginRegistrationSurface",
    "bootstrap_harness_host_application_plugins",
    "bootstrap_harness_host_platform",
    "resolve_harness_host_decision_flow_gate",
    "resolve_harness_host_execution_budget_ledger_factory",
    "resolve_harness_host_event_bus",
    "resolve_harness_host_execution_terminal",
    "resolve_harness_host_lifecycle_hook_coordinator",
    "resolve_harness_host_middleware_pipeline",
    "resolve_harness_host_plugin_bootstrap_target",
    "resolve_harness_host_orchestration_topology_submission_port",
    "resolve_harness_host_runtime_event_persistence",
]
