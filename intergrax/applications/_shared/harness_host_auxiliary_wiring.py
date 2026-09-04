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
from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
from intergrax.applications._shared.platform_wiring import bootstrap_nexus_platform
from intergrax.applications._shared.plugin_bootstrap import PluginBootstrapResult
from intergrax.applications._shared.task_control_wiring import (
    TaskEnricher,
    build_task_runner_with_enricher,
)
from intergrax.debug.app import create_debug_app
from intergrax.debug.hitl_service import DebugHitlResumeService
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.interactions.intake_service import InteractionIntakeService
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

if TYPE_CHECKING:
    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime


class HostTaskExecutionExecutor:
    """Execute interaction-intake tasks through canonical host task execution."""

    def __init__(self, host_execution: HostTaskExecutionPort) -> None:
        self._host_execution = host_execution

    async def execute(self, task: Task) -> TaskResult:
        return await self._host_execution.execute(task)


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
        task_executor=HostTaskExecutionExecutor(host_execution),
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
]
