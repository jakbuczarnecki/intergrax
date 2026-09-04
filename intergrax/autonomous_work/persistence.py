# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral Autonomous Work repository bundle contracts (AW-2C)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.autonomous_work.repository import (
    GoalEvaluationCadenceStateRepository,
    ResponsibilityRepository,
    WorkContinuityStateRepository,
    WorkerDefinitionRepository,
    WorkerGoalRepository,
    WorkerInstanceRepository,
    WorkerPrincipalBindingRepository,
    WorkerWakeUpReceiptRepository,
)
from intergrax.autonomous_work.worker_budget_ports import WorkerAccountingRepository


@runtime_checkable
class AutonomousWorkStoreOwner(Protocol):
    """Lifecycle owner for durable Autonomous Work repository adapters."""

    def close(self) -> None:
        """Release persistence resources."""


@dataclass(frozen=True, slots=True)
class AutonomousWorkRepositories:
    """Bundle of authoritative Autonomous Work repository ports."""

    worker_definition: WorkerDefinitionRepository
    worker_instance: WorkerInstanceRepository
    responsibility: ResponsibilityRepository
    worker_goal: WorkerGoalRepository
    work_continuity_state: WorkContinuityStateRepository
    goal_evaluation_cadence_state: GoalEvaluationCadenceStateRepository
    worker_principal_binding: WorkerPrincipalBindingRepository
    worker_wake_up_receipt: WorkerWakeUpReceiptRepository
    worker_accounting: WorkerAccountingRepository
    store: AutonomousWorkStoreOwner

    def close(self) -> None:
        self.store.close()
