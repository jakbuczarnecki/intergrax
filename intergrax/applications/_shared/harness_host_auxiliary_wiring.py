# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime-scoped auxiliary wiring for Tier-3 harness hosts (NPSC-2).

Author-facing application code should depend on ``HarnessHostRuntime.execution``.
Internal platform subsystems (plugins, scheduler, debug API) compose through
typed harness host capabilities — not in generated Tier-3 hosts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI

from intergrax.applications._shared.harness_host_composition import (
    bootstrap_harness_host_platform,
    resolve_harness_host_execution_terminal,
)
from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
from intergrax.applications._shared.task_control_wiring import (
    TaskEnricher,
    wire_harness_task_control,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.debug.app import create_debug_app
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.interactions.intake_service import InteractionIntakeService
from intergrax.runtime.interactions.task_executor import HostTaskExecutionExecutor
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.long_running.wiring import (
    LongRunningSchedulerWiring,
    wire_long_running_scheduler_with_host_execution,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry

if TYPE_CHECKING:
    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime


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
    return wire_long_running_scheduler_with_host_execution(
        checkpoint_store=checkpoint_store,
        host_execution=host_execution,
        execution_terminal=resolve_harness_host_execution_terminal(runtime),
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
    return wire_harness_task_control(
        app,
        enabled=enabled,
        host_execution=host_execution,
        env=env,
        checkpoint_store=checkpoint_store,
        task_route_prefix=task_route_prefix,
        extra_enricher=extra_enricher,
        task_enricher=task_enricher,
        runtime=runtime,
    )


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
        host_execution=runtime.execution,
        interaction_service=interaction_service,
        hitl_service=hitl_service,
        checkpoint_store=checkpoint_store,
        trace_store=runtime.observability.trace_store,
        runtime_event_store=runtime.observability.runtime_event_store,
    )


__all__ = [
    "HostTaskExecutionExecutor",
    "bootstrap_harness_host_platform",
    "create_harness_host_debug_app",
    "wire_harness_host_interaction_intake",
    "wire_harness_host_long_running_scheduler",
    "wire_harness_host_task_control",
]
