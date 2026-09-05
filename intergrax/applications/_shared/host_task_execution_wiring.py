# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared composition-root host task execution wiring for application hosts."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.profile_resolution.execution_admission import (
    EffectiveProfileExecutionPinningDependencies,
    EffectiveProfileRevisionAdmission,
    build_effective_profile_revision_admission,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId
from intergrax.runtime.execution.effective_profile_revision_admission import (
    EffectiveProfileRevisionAdmissionPort,
)
from intergrax.runtime.execution.host_task import HostTaskExecution
from intergrax.runtime.execution.host_task_terminal_publisher import HostTaskTerminalPublisher
from intergrax.runtime.execution.orchestration import OrchestrationExecutor
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.orchestration_capabilities import orchestration_capabilities_from_triggers
from intergrax.runtime.task.task import Task


@dataclass(frozen=True, slots=True)
class _NexusHostTaskTerminalPublisher:
    _nexus_loop: NexusLoop

    async def publish_terminal(
        self,
        task: Task,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
    ) -> None:
        await self._nexus_loop.publish_host_task_terminal_runtime(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )


def build_nexus_host_task_terminal_publisher(
    nexus_loop: NexusLoop,
) -> HostTaskTerminalPublisher:
    """Internal composition adapter: delegate terminal publication to Nexus."""
    return _NexusHostTaskTerminalPublisher(_nexus_loop=nexus_loop)


def build_host_task_execution(
    nexus_loop: NexusLoop,
    *,
    orchestration_triggers: frozenset[str],
    pipeline_capability_suffix: str = ".pipeline",
    revision_admission: EffectiveProfileRevisionAdmissionPort | None = None,
) -> HostTaskExecution:
    """Internal composition builder: extract canonical execution dependencies from Nexus."""
    return HostTaskExecution(
        _agent_engine=nexus_loop.agent_engine,
        _agent_router=AgentRouter(
            nexus_loop.registry,
            event_bus=nexus_loop.event_bus,
        ),
        _orchestration_executor=OrchestrationExecutor(nexus_loop),
        _orchestration_triggers=orchestration_triggers,
        _pipeline_capability_suffix=pipeline_capability_suffix,
        _ledger_factory=nexus_loop.execution_budget_ledger_factory,
        _run_budget=nexus_loop.run_budget,
        _terminal_publisher=build_nexus_host_task_terminal_publisher(nexus_loop),
        _revision_admission=revision_admission,
    )


def build_environment_host_task_execution(
    nexus_loop: NexusLoop,
    env: ApplicationEnvironmentProfile,
    *,
    pinning_dependencies: EffectiveProfileExecutionPinningDependencies | None = None,
) -> HostTaskExecution:
    """Build canonical host task execution from environment orchestration profile."""
    graph_spec = env.graph_spec
    revision_admission = (
        build_effective_profile_revision_admission(pinning_dependencies)
        if pinning_dependencies is not None
        else None
    )
    return build_host_task_execution(
        nexus_loop,
        orchestration_triggers=orchestration_capabilities_from_triggers(
            graph_spec.trigger_capabilities if graph_spec is not None else None,
        ),
        pipeline_capability_suffix=(
            graph_spec.pipeline_capability_suffix if graph_spec is not None else ".pipeline"
        ),
        revision_admission=revision_admission,
    )
