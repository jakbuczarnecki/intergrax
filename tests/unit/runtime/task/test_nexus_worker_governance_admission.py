# © Artur Czarnecki. All rights reserved.

"""Queue worker governance admission — explicit trusted boundary (P2C-R0A-R1-R2)."""

from __future__ import annotations

import inspect

import pytest
from echo.echo_agent import EchoAgent

from intergrax.contracts.admitted_root_governance_identity import (
    AdmittedRootGovernanceIdentity,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.nexus_worker_execution import NexusWorkerRuntime
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.worker_bootstrap import build_nexus_task_execution_registry
from testing_support.admitted_root_governance_identity import (
    lab_admitted_root_governance_identity_for_task,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_from_registry_rejects_missing_governance_admission() -> None:
    signature = inspect.signature(NexusWorkerRuntime.from_registry)
    assert (
        signature.parameters["admit_root_governance_identity"].default
        is inspect.Parameter.empty
    )
    assert signature.parameters["production_mode"].default is inspect.Parameter.empty


def test_build_nexus_task_execution_registry_requires_trusted_admission() -> None:
    with pytest.raises(ValueError, match="host_execution or"):
        build_nexus_task_execution_registry(AgentRegistry(), production_mode=False)


def test_explicit_admission_binds_governance_identity_on_execute() -> None:
    bound: list[AdmittedRootGovernanceIdentity] = []

    def _admit(task: Task) -> AdmittedRootGovernanceIdentity:
        admitted = lab_admitted_root_governance_identity_for_task(task)
        bound.append(admitted)
        return admitted

    registry = AgentRegistry()
    registry.register(EchoAgent())
    runtime = NexusWorkerRuntime.from_registry(
        registry,
        production_mode=False,
        admit_root_governance_identity=_admit,
    )
    assert runtime.host_execution is not None


def test_admission_rejects_task_without_principal() -> None:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    runtime = NexusWorkerRuntime.from_registry(
        registry,
        production_mode=False,
        admit_root_governance_identity=lab_admitted_root_governance_identity_for_task,
    )
    task = Task(
        tenant_id="tenant-a",
        user_id="",
        message="no principal",
        context=TaskContext(capability="echo.basic"),
    )
    with pytest.raises(ValueError, match="tenant_id and user_id"):
        lab_admitted_root_governance_identity_for_task(task)
    assert runtime.host_execution is not None
