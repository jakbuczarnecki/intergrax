# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read ports for durable recovery fulfillment episode projection (UCA-6C-R6-R5.8-R2-H1-R1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.autonomous_work.capability_acquisition import WorkerCapabilityNeed
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId


@dataclass(frozen=True, slots=True)
class WorkerRecoveryFulfillmentTaskContext:
    """Canonical task/tenant correlation for recovery fulfillment — read-only facts."""

    task_id: TaskId
    tenant_id: str
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None


class WorkerRecoveryObstacleCapabilityNeedReadPort(Protocol):
    """Load durable worker capability need recorded for an obstacle episode."""

    def get_obstacle_capability_need(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
        obstacle_id: str,
    ) -> WorkerCapabilityNeed | None: ...


class WorkerRecoveryFulfillmentTaskContextReadPort(Protocol):
    """Resolve active execution task context without discovery or acquisition."""

    def resolve_task_context(
        self,
        *,
        run_id: RunId | None,
    ) -> WorkerRecoveryFulfillmentTaskContext | None: ...


__all__ = [
    "WorkerRecoveryFulfillmentTaskContext",
    "WorkerRecoveryFulfillmentTaskContextReadPort",
    "WorkerRecoveryObstacleCapabilityNeedReadPort",
]
