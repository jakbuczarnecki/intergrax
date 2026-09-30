# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Certified harness root execution port — Nexus stays inside Execution Engine."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.contracts.admitted_root_governance_identity import AdmittedRootGovernanceIdentity
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId
from intergrax.contracts.run_budget import RunBudget
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedgerFactory
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.execution.orchestration import execute_root_task, resolve_root_task_identity
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationApplicationWiringTarget,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskResult


class _PerTaskGovernanceHarnessPort:
    __slots__ = ("_admit", "_harness")

    def __init__(
        self,
        harness: HarnessRootTaskExecutionPort,
        admit: Callable[[Task], AdmittedRootGovernanceIdentity],
    ) -> None:
        self._harness = harness
        self._admit = admit

    async def execute(
        self,
        task: Task,
        *,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
        execution_id: ExecutionId | None = None,
        resume_checkpoint: TaskCheckpoint | None = None,
        restore_existing_execution: bool = False,
    ) -> TaskResult:
        return await self._harness.execute(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            resume_checkpoint=resume_checkpoint,
            restore_existing_execution=restore_existing_execution,
            admitted_governance_identity=self._admit(task),
        )


class HarnessRootTaskExecutionPort:
    """Thin :class:`HostTaskExecutionPort` over internal ``execute_root_task``."""

    __slots__ = (
        "_admit_root_governance_identity",
        "_ledger_factory",
        "_nexus_loop",
        "_run_budget",
    )

    def __init__(
        self,
        nexus_loop: NexusLoop,
        *,
        execution_budget_ledger_factory: ExecutionBudgetLedgerFactory | None = None,
        run_budget: RunBudget | None = None,
        admit_root_governance_identity: (
            Callable[[Task], AdmittedRootGovernanceIdentity] | None
        ) = None,
    ) -> None:
        self._nexus_loop = nexus_loop
        self._ledger_factory = (
            execution_budget_ledger_factory
            or nexus_loop.execution_budget_ledger_factory
        )
        self._run_budget = run_budget if run_budget is not None else nexus_loop.run_budget
        self._admit_root_governance_identity = admit_root_governance_identity

    def with_per_task_governance_admission(
        self,
        admit: Callable[[Task], AdmittedRootGovernanceIdentity],
    ) -> HostTaskExecutionPort:
        return _PerTaskGovernanceHarnessPort(self, admit)

    async def execute(
        self,
        task: Task,
        *,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
        execution_id: ExecutionId | None = None,
        resume_checkpoint: TaskCheckpoint | None = None,
        restore_existing_execution: bool = False,
        admitted_governance_identity: AdmittedRootGovernanceIdentity | None = None,
    ) -> TaskResult:
        del execution_id, restore_existing_execution
        identity = resolve_root_task_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            resume_checkpoint=resume_checkpoint,
        )
        admitted = admitted_governance_identity
        if admitted is None and self._admit_root_governance_identity is not None:
            admitted = self._admit_root_governance_identity(task)
        return await execute_root_task(
            task,
            nexus_loop=self._nexus_loop,
            identity=identity,
            admitted_governance_identity=admitted,
            resume_checkpoint=resume_checkpoint,
            ledger_factory=self._ledger_factory,
            run_budget=self._run_budget,
        )


def build_harness_root_task_execution_port_from_wiring_target(
    host: HostOrchestrationApplicationWiringTarget,
    *,
    execution_budget_ledger_factory: ExecutionBudgetLedgerFactory | None = None,
    run_budget: RunBudget | None = None,
    admit_root_governance_identity: (
        Callable[[Task], AdmittedRootGovernanceIdentity] | None
    ) = None,
) -> HostTaskExecutionPort:
    if not isinstance(host, NexusLoop):
        raise TypeError(
            "harness root task execution requires an internal orchestration host materialization",
        )
    return build_harness_root_task_execution_port(
        host,
        execution_budget_ledger_factory=execution_budget_ledger_factory,
        run_budget=run_budget,
        admit_root_governance_identity=admit_root_governance_identity,
    )


def build_harness_root_task_execution_port(
    nexus_loop: NexusLoop,
    *,
    execution_budget_ledger_factory: ExecutionBudgetLedgerFactory | None = None,
    run_budget: RunBudget | None = None,
    admit_root_governance_identity: (
        Callable[[Task], AdmittedRootGovernanceIdentity] | None
    ) = None,
) -> HostTaskExecutionPort:
    return HarnessRootTaskExecutionPort(
        nexus_loop,
        execution_budget_ledger_factory=execution_budget_ledger_factory,
        run_budget=run_budget,
        admit_root_governance_identity=admit_root_governance_identity,
    )


__all__ = [
    "HarnessRootTaskExecutionPort",
    "build_harness_root_task_execution_port",
]
