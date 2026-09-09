# © Artur Czarnecki. All rights reserved.

"""Shared NPSC-5D governance test adapters."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

from intergrax.runtime.governance.active_governed_execution_task import (
    bind_governed_execution_task,
    reset_governed_execution_task,
)
from intergrax.runtime.governance.multi_agent_coordination_governance import (
    AllowingMultiAgentCoordinationGovernance,
    DenyingMultiAgentCoordinationGovernance,
    RequireHumanMultiAgentCoordinationGovernance,
    UnavailableMultiAgentCoordinationGovernance,
)
from intergrax.runtime.governance.physical_delegation_governance import (
    AllowingPhysicalDelegationGovernance,
    DenyingPhysicalDelegationGovernance,
    RequireHumanPhysicalDelegationGovernance,
    UnavailablePhysicalDelegationGovernance,
)
from intergrax.runtime.task.task import Task


def non_collaborative_governed_host_task(
    *,
    tenant_id: str = "tenant-a",
    user_id: str = "admin-user",
    agent_id: str = "agent-a",
) -> Task:
    return Task(
        tenant_id=tenant_id,
        user_id=user_id,
        agent_id=agent_id,
        metadata={},
    )


@contextmanager
def bound_governed_host_task(task: Task | None = None) -> Iterator[Task]:
    resolved = task or non_collaborative_governed_host_task()
    token = bind_governed_execution_task(resolved)
    try:
        yield resolved
    finally:
        reset_governed_execution_task(token)


def allowing_coordination_governance() -> AllowingMultiAgentCoordinationGovernance:
    return AllowingMultiAgentCoordinationGovernance()


def denying_coordination_governance() -> DenyingMultiAgentCoordinationGovernance:
    return DenyingMultiAgentCoordinationGovernance()


def require_human_coordination_governance() -> RequireHumanMultiAgentCoordinationGovernance:
    return RequireHumanMultiAgentCoordinationGovernance()


def unavailable_coordination_governance() -> UnavailableMultiAgentCoordinationGovernance:
    return UnavailableMultiAgentCoordinationGovernance()


def allowing_physical_delegation_governance() -> AllowingPhysicalDelegationGovernance:
    return AllowingPhysicalDelegationGovernance()


def denying_physical_delegation_governance() -> DenyingPhysicalDelegationGovernance:
    return DenyingPhysicalDelegationGovernance()


def require_human_physical_delegation_governance() -> RequireHumanPhysicalDelegationGovernance:
    return RequireHumanPhysicalDelegationGovernance()


def unavailable_physical_delegation_governance() -> UnavailablePhysicalDelegationGovernance:
    return UnavailablePhysicalDelegationGovernance()
