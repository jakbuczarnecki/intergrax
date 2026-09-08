# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime-scoped auxiliary wiring for Tier-3 harness hosts (NPSC-2).

Author-facing application code should depend on ``HarnessHostRuntime.execution``.
Internal platform subsystems (plugins, scheduler task runner, debug API) still
compose through legacy Nexus handles resolved here — not in generated Tier-3 hosts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI

from intergrax.applications._shared.harness_host_runtime_compat import (
    resolve_harness_host_nexus_loop_legacy,
)
from intergrax.applications._shared.harness_task_routes import mount_canonical_harness_task_routes
from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.plugin_bootstrap import PluginBootstrapResult
from intergrax.applications._shared.task_control_wiring import (
    TaskEnricher,
    build_reliability_task_enricher,
    build_task_runner_with_enricher,
    resolve_harness_task_control_execution_terminal,
)
from intergrax.applications._shared.async_task_index_resolver import resolve_async_task_index
from intergrax.applications._shared.harness_control_plane_governance_wiring import (
    resolve_harness_task_control_mutation_boundary,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.long_running.wiring import (
    LongRunningSchedulerWiring,
    wire_long_running_scheduler_with_host_execution,
)
from intergrax.debug.app import create_debug_app
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.runtime.interactions.task_executor import HostTaskExecutionExecutor
from intergrax.runtime.interactions.intake_service import InteractionIntakeService
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

if TYPE_CHECKING:
    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime


def bootstrap_harness_host_platform(runtime: HarnessHostRuntime) -> PluginBootstrapResult:
    """Register default runtime plugins for a composed harness host."""
    nexus_loop = resolve_harness_host_nexus_loop_legacy(runtime)
    return bootstrap_nexus_platform(
        nexus_loop,
        trace_store=runtime.observability.trace_store,
    )


def build_harness_host_task_runner(
    runtime: HarnessHostRuntime,
    *,
    enricher: TaskEnricher | None = None,
) -> UnifiedTaskRunner:
    """Background scheduler / task-control runner bound to the host runtime."""
    return build_task_runner_with_enricher(
        resolve_harness_host_nexus_loop_legacy(runtime),
        enricher,
    )


def wire_harness_host_long_running_scheduler(
    runtime: HarnessHostRuntime,
    *,
    checkpoint_store: TaskCheckpointPersistence,
    host_execution: HostTaskExecutionPort,
    task_enricher: TaskEnricher | None = None,
    notification_adapter=None,
    poll_interval_seconds: float | None = None,
    enabled: bool = True,
) -> LongRunningSchedulerWiring | None:
    """Long-running scheduler wired to canonical host execution."""
    nexus_loop = resolve_harness_host_nexus_loop_legacy(runtime)
    return wire_long_running_scheduler_with_host_execution(
        checkpoint_store=checkpoint_store,
        host_execution=host_execution,
        execution_terminal=nexus_loop.execution_terminal,
        task_enricher=task_enricher,
        notification_adapter=notification_adapter,
        poll_interval_seconds=poll_interval_seconds,
        enabled=enabled,
    )


def wire_harness_host_task_control(
    app: FastAPI,
    *,
    enabled: bool,
    host_execution: HostTaskExecutionPort,
    env: ApplicationEnvironmentProfile,
    runtime: HarnessHostRuntime,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    task_route_prefix: str = "/v1/tasks",
    extra_enricher: TaskEnricher | None = None,
    task_enricher: TaskEnricher | None = None,
) -> TaskEnricher:
    """Mount harness task HTTP routes through canonical host execution."""
    enricher = task_enricher or build_reliability_task_enricher(env, extra=extra_enricher)
    task_executor = HostTaskExecutionExecutor(host_execution, task_enricher=enricher)
    resolved_boundary = None
    if runtime.control_plane_governance is not None:
        resolved_boundary = resolve_harness_task_control_mutation_boundary(
            runtime.control_plane_governance,
        )
    if enabled:
        async_index = resolve_async_task_index(env)
        resolved_terminal = resolve_harness_task_control_execution_terminal(runtime=runtime)
        mount_canonical_harness_task_routes(
            app,
            task_executor=task_executor,
            host_execution=host_execution,
            checkpoint_store=checkpoint_store,
            execution_terminal=resolved_terminal,
            prefix=task_route_prefix,
            task_enricher=enricher,
            async_index=async_index,
            mutation_boundary=resolved_boundary,
        )
    return enricher


def wire_harness_host_interaction_intake(
    runtime: HarnessHostRuntime,
    *,
    host_execution: HostTaskExecutionPort,
    interaction_surface: str = "auto",
    task_enricher: TaskEnricher | None = None,
) -> InteractionIntakeService:
    """Inbound interaction intake routed through canonical host execution."""
    _ = runtime
    return wire_interaction_intake_service(
        task_executor=HostTaskExecutionExecutor(host_execution, task_enricher=task_enricher),
        interaction_surface=interaction_surface,
        task_enricher=task_enricher,
    )


def create_harness_host_debug_app(
    runtime: HarnessHostRuntime,
    *,
    registry: AgentRegistry,
    experiments_db_path: Path | None = None,
    checkpoints_db_path: Path | None = None,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    interaction_service: InteractionIntakeService | None = None,
    hitl_service: DebugHitlResumeService | None = None,
) -> FastAPI:
    """Laboratory debug API over the composed harness host runtime."""
    return create_debug_app(
        db_path=runtime.observability.trace_db_path,
        experiments_db_path=experiments_db_path,
        runtime_events_db_path=runtime.observability.runtime_events_db_path,
        checkpoints_db_path=checkpoints_db_path,
        registry=registry,
        nexus_loop=resolve_harness_host_nexus_loop_legacy(runtime),
        interaction_service=interaction_service,
        hitl_service=hitl_service,
        checkpoint_store=checkpoint_store,
        trace_store=runtime.observability.trace_store,
        runtime_event_store=runtime.observability.runtime_event_store,
    )


__all__ = [
    "HostTaskExecutionExecutor",
    "bootstrap_harness_host_platform",
    "build_harness_host_task_runner",
    "create_harness_host_debug_app",
    "wire_harness_host_interaction_intake",
    "wire_harness_host_long_running_scheduler",
    "wire_harness_host_task_control",
]
