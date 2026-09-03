# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker wake-up admission service (AW-4A).

Accepts typed wake-up signals, enforces lifecycle/source eligibility, durable
idempotency, and restores orientation from WorkContinuityState.

Does not dispatch work, call LLM, create Execution, or mutate lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from intergrax.autonomous_work.lifecycle import AutonomousWorkClock
from intergrax.autonomous_work.repository import (
    AutonomousWorkEntityNotFound,
    WorkContinuityStateRepository,
    WorkerInstanceRepository,
    WorkerWakeUpReceiptRepository,
)
from intergrax.contracts.autonomous_work.continuity import WorkContinuityState
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId, validate_worker_instance_id
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.wake_up import (
    WorkerWakeUpContext,
    WorkerWakeUpDisposition,
    WorkerWakeUpReceipt,
    WorkerWakeUpResult,
    WorkerWakeUpSignal,
    WorkerWakeUpSourceKind,
)
from intergrax.contracts.autonomous_work.worker import WorkerInstance

_ORDINARY_WAKE_UP_SOURCES: Final[frozenset[WorkerWakeUpSourceKind]] = frozenset(
    {
        WorkerWakeUpSourceKind.EXTERNAL_EVENT,
        WorkerWakeUpSourceKind.QUEUE_DELIVERY,
        WorkerWakeUpSourceKind.SCHEDULE,
        WorkerWakeUpSourceKind.OPERATOR,
    }
)
_RECOVERY_WAKE_UP_SOURCES: Final[frozenset[WorkerWakeUpSourceKind]] = frozenset(
    {
        WorkerWakeUpSourceKind.DEPENDENCY_RECOVERY,
        WorkerWakeUpSourceKind.RECOVERY_TIMER,
    }
)
_HUMAN_WAKE_UP_SOURCES: Final[frozenset[WorkerWakeUpSourceKind]] = frozenset(
    {WorkerWakeUpSourceKind.HUMAN_CONTINUATION}
)
_OPERATIONAL_LIFECYCLE_STATES: Final[frozenset[WorkerLifecycleState]] = frozenset(
    {
        WorkerLifecycleState.ACTIVE,
        WorkerLifecycleState.IDLE,
        WorkerLifecycleState.WORKING,
        WorkerLifecycleState.RECOVERING,
        WorkerLifecycleState.DEGRADED,
    }
)


class WorkerWakeUpPersistenceUnavailable(Exception):
    """Durable wake-up receipt store is unavailable — admission fails closed."""


@dataclass(frozen=True, slots=True)
class WorkerWakeUpEligibility:
    """Lifecycle/source eligibility decision for one wake-up attempt."""

    disposition: WorkerWakeUpDisposition
    eligible: bool


class WorkerWakeUpEligibilityPolicy:
    """Pure canonical wake-up eligibility policy — no lifecycle mutation."""

    def evaluate(
        self,
        *,
        lifecycle_state: WorkerLifecycleState,
        source_kind: WorkerWakeUpSourceKind,
    ) -> WorkerWakeUpEligibility:
        if lifecycle_state in {
            WorkerLifecycleState.STOPPED,
            WorkerLifecycleState.QUARANTINED,
        }:
            return WorkerWakeUpEligibility(
                disposition=WorkerWakeUpDisposition.REJECTED,
                eligible=False,
            )
        if lifecycle_state in {
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.PROVISIONING,
        }:
            return WorkerWakeUpEligibility(
                disposition=WorkerWakeUpDisposition.NOT_ELIGIBLE,
                eligible=False,
            )
        if lifecycle_state == WorkerLifecycleState.WAITING_EXTERNAL:
            if source_kind in _RECOVERY_WAKE_UP_SOURCES:
                return WorkerWakeUpEligibility(
                    disposition=WorkerWakeUpDisposition.ACCEPTED,
                    eligible=True,
                )
            return WorkerWakeUpEligibility(
                disposition=WorkerWakeUpDisposition.NOT_ELIGIBLE,
                eligible=False,
            )
        if lifecycle_state == WorkerLifecycleState.WAITING_FOR_HUMAN:
            if source_kind in _HUMAN_WAKE_UP_SOURCES:
                return WorkerWakeUpEligibility(
                    disposition=WorkerWakeUpDisposition.ACCEPTED,
                    eligible=True,
                )
            return WorkerWakeUpEligibility(
                disposition=WorkerWakeUpDisposition.NOT_ELIGIBLE,
                eligible=False,
            )
        if lifecycle_state in _OPERATIONAL_LIFECYCLE_STATES:
            allowed = (
                _ORDINARY_WAKE_UP_SOURCES
                | _RECOVERY_WAKE_UP_SOURCES
                | _HUMAN_WAKE_UP_SOURCES
            )
            if source_kind in allowed:
                return WorkerWakeUpEligibility(
                    disposition=WorkerWakeUpDisposition.ACCEPTED,
                    eligible=True,
                )
        return WorkerWakeUpEligibility(
            disposition=WorkerWakeUpDisposition.NOT_ELIGIBLE,
            eligible=False,
        )


class WorkerWakeUpService:
    """Authoritative wake-up admission without work dispatch or lifecycle mutation."""

    def __init__(
        self,
        *,
        worker_instance_repository: WorkerInstanceRepository,
        continuity_state_repository: WorkContinuityStateRepository,
        wake_up_receipt_repository: WorkerWakeUpReceiptRepository,
        clock: AutonomousWorkClock | Callable[[], datetime],
        eligibility_policy: WorkerWakeUpEligibilityPolicy | None = None,
    ) -> None:
        self._worker_repository = worker_instance_repository
        self._continuity_repository = continuity_state_repository
        self._receipt_repository = wake_up_receipt_repository
        self._clock = clock
        self._eligibility_policy = eligibility_policy or WorkerWakeUpEligibilityPolicy()

    def accept(self, signal: WorkerWakeUpSignal) -> WorkerWakeUpResult:
        """Accept one wake-up delivery with durable idempotency and orientation restore."""
        if type(signal) is not WorkerWakeUpSignal:
            raise TypeError("signal must be WorkerWakeUpSignal")
        worker = self._load_worker(worker_instance_id=signal.worker_instance_id)
        eligibility = self._eligibility_policy.evaluate(
            lifecycle_state=worker.lifecycle_state,
            source_kind=signal.source_kind,
        )
        if not eligibility.eligible:
            return self._result(
                disposition=eligibility.disposition,
                signal=signal,
                worker=worker,
                continuity_state=self._load_continuity(worker_instance_id=worker.worker_instance_id),
                accepted_at=self._now(),
                receipt=None,
            )
        receipt = WorkerWakeUpReceipt(
            worker_instance_id=signal.worker_instance_id,
            wake_up_id=signal.wake_up_id,
            source_kind=signal.source_kind,
            source_ref=signal.source_ref,
            occurred_at=signal.occurred_at,
            accepted_at=self._now(),
            delivery_identity=signal.delivery_identity,
            correlation_ref=signal.correlation_ref,
        )
        claim = self._receipt_repository.claim(receipt)
        continuity_state = self._load_continuity(
            worker_instance_id=worker.worker_instance_id,
        )
        if claim.duplicate:
            return self._result(
                disposition=WorkerWakeUpDisposition.DUPLICATE,
                signal=signal,
                worker=worker,
                continuity_state=continuity_state,
                accepted_at=claim.receipt.accepted_at,
                receipt=claim.receipt,
            )
        return self._result(
            disposition=WorkerWakeUpDisposition.ACCEPTED,
            signal=signal,
            worker=worker,
            continuity_state=continuity_state,
            accepted_at=claim.receipt.accepted_at,
            receipt=claim.receipt,
        )

    def _load_worker(self, *, worker_instance_id: WorkerInstanceId) -> WorkerInstance:
        validate_worker_instance_id(worker_instance_id)
        worker = self._worker_repository.get(worker_instance_id=worker_instance_id)
        if worker is None:
            raise AutonomousWorkEntityNotFound(
                f"WorkerInstance not found for {worker_instance_id}"
            )
        return worker

    def _load_continuity(
        self,
        *,
        worker_instance_id: WorkerInstanceId,
    ) -> WorkContinuityState | None:
        return self._continuity_repository.get(worker_instance_id=worker_instance_id)

    def _result(
        self,
        *,
        disposition: WorkerWakeUpDisposition,
        signal: WorkerWakeUpSignal,
        worker: WorkerInstance,
        continuity_state: WorkContinuityState | None,
        accepted_at: datetime,
        receipt: WorkerWakeUpReceipt | None,
    ) -> WorkerWakeUpResult:
        context = WorkerWakeUpContext(
            worker_instance=worker,
            wake_up_signal=signal,
            continuity_state=continuity_state,
            accepted_at=accepted_at,
            disposition=disposition,
            receipt=receipt,
        )
        return WorkerWakeUpResult(disposition=disposition, context=context)

    def _now(self) -> datetime:
        if isinstance(self._clock, AutonomousWorkClock):
            return self._clock.now()
        return self._clock()
