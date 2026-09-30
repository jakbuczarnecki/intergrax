# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compose application host orchestration through Execution Engine only."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionPort,
)
from intergrax.runtime.execution._orchestration_backend_access import (
    orchestration_backend_for_execution_engine,
)
from intergrax.runtime.execution.application_host_orchestration_session import (
    ApplicationHostOrchestrationSession,
    HostOrchestrationTraceLifecycleControl,
)
from intergrax.runtime.execution.environment_host_task_execution import (
    build_environment_host_task_execution,
)
from intergrax.runtime.execution.environment_orchestration_materialization import (
    EnvironmentOrchestrationMaterialization,
    materialize_host_orchestration_backend,
)
from intergrax.runtime.execution.host_orchestration_loop_init_spec import (
    HostOrchestrationLoopInitSpec,
)
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.task.task import Task


def compose_application_host_orchestration_session(
    registry: AgentRegistryRead,
    spec: HostOrchestrationLoopInitSpec,
    env: ApplicationEnvironmentProfile,
    *,
    root_authority_admission: RootExecutionAuthorityAdmissionPort,
    admit_root_governance_identity: Callable[[Task], AdmittedRootGovernanceIdentity],
    orchestration_triggers: frozenset[str] | None = None,
    pipeline_capability_suffix: str | None = None,
) -> tuple[ApplicationHostOrchestrationSession, EnvironmentOrchestrationMaterialization]:
    """
    Materialize private orchestration backend and return neutral host session.

    ``EnvironmentOrchestrationMaterialization`` is returned only for EE-internal
    harness wiring; Tier-3 scenario code must consume ``ApplicationHostOrchestrationSession``.
    """
    materialization = materialize_host_orchestration_backend(registry, spec)
    backend = orchestration_backend_for_execution_engine(materialization)
    host_execution = build_environment_host_task_execution(
        materialization,
        env,
        orchestration_triggers=orchestration_triggers,
        pipeline_capability_suffix=pipeline_capability_suffix,
        root_authority_admission=root_authority_admission,
        admit_root_governance_identity=admit_root_governance_identity,
    )
    session = ApplicationHostOrchestrationSession(
        host_execution=host_execution,
        runtime_event_bus=materialization.event_bus,
        trace_lifecycle=HostOrchestrationTraceLifecycleControl(backend),
    )
    return session, materialization


__all__ = ["compose_application_host_orchestration_session"]
