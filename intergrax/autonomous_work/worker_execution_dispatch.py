# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Worker execution dispatch service (AW-5A).

Orchestrates worker eligibility, AW-3B collaborative authority admission,
Runtime/Governance trusted root authority admission, and canonical execution
intake. Does not own Run/Attempt/Execution lifecycle or mint trusted authority.

``DISPATCHED`` means canonical execution invocation completed successfully — not
asynchronous acceptance. AW-local correlation is not durable source of truth;
canonical runtime IDs/events are the durable execution evidence. Same ``RunId``
is correlation only — not idempotent when ExecutionRuntime mints a new
``ExecutionId`` per invocation.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final, Generic, TypeVar

from intergrax.autonomous_work.execution_authority_admission import (
    WorkerExecutionAdmissionService,
    WorkerExecutionAuthorityDenied,
)
from intergrax.autonomous_work.repository import (
    ResponsibilityRepository,
    WorkerGoalRepository,
    WorkerInstanceRepository,
)
from intergrax.contracts.autonomous_work.execution_authority import (
    WorkerExecutionAuthorityRequest,
)
from intergrax.contracts.autonomous_work.execution_dispatch import (
    WorkerExecutionCorrelation,
    WorkerExecutionDispatchDisposition,
    WorkerExecutionDispatchRejectionReason,
    WorkerExecutionDispatchRequest,
    WorkerExecutionDispatchResult,
)
from intergrax.contracts.autonomous_work.goal import WorkerGoalStatus
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.responsibility import ResponsibilityStatus
from intergrax.contracts.execution_intake import (
    CanonicalExecutionIntakePort,
    CanonicalExecutionIntakeRequest,
    CanonicalExecutionInvocationFailed,
)
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionDisposition,
    RootExecutionAuthorityAdmissionPort,
    RootExecutionAuthorityAdmissionRequest,
)

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

_INELIGIBLE_WORKER_LIFECYCLE_STATES: Final[frozenset[WorkerLifecycleState]] = frozenset(
    {
        WorkerLifecycleState.PAUSED,
        WorkerLifecycleState.QUARANTINED,
        WorkerLifecycleState.STOPPED,
        WorkerLifecycleState.PROVISIONING,
    }
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class WorkerExecutionDispatchService(Generic[InputT, OutputT]):
    """Canonical Worker → Execution dispatch with fail-closed admission gates."""

    def __init__(
        self,
        *,
        worker_instance_repository: WorkerInstanceRepository,
        responsibility_repository: ResponsibilityRepository,
        worker_goal_repository: WorkerGoalRepository,
        admission_service: WorkerExecutionAdmissionService,
        root_authority_admission: RootExecutionAuthorityAdmissionPort,
        execution_intake: CanonicalExecutionIntakePort[InputT, OutputT],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._worker_instance_repository = worker_instance_repository
        self._responsibility_repository = responsibility_repository
        self._worker_goal_repository = worker_goal_repository
        self._admission_service = admission_service
        self._root_authority_admission = root_authority_admission
        self._execution_intake = execution_intake
        self._clock = clock or _utc_now

    async def dispatch(
        self,
        request: WorkerExecutionDispatchRequest[InputT, OutputT],
    ) -> WorkerExecutionDispatchResult[OutputT]:
        if type(request) is not WorkerExecutionDispatchRequest:
            raise TypeError("request must be WorkerExecutionDispatchRequest")

        base_correlation = WorkerExecutionCorrelation(
            worker_instance_id=request.worker_instance_id,
            source=request.source,
            run_id=None,
            attempt_id=None,
            execution_id=None,
            goal_id=request.goal_id,
            responsibility_id=request.responsibility_id,
            wake_up_id=request.wake_up_id,
            collaborative_work_ref=request.collaborative_work_ref,
            created_at=self._clock(),
        )

        worker = self._worker_instance_repository.get(
            worker_instance_id=request.worker_instance_id,
        )
        if worker is None:
            return self._rejected(
                correlation=base_correlation,
                reason=WorkerExecutionDispatchRejectionReason.OWNERSHIP_MISMATCH,
            )
        if worker.revision != request.worker_revision:
            return self._rejected(
                correlation=base_correlation,
                reason=WorkerExecutionDispatchRejectionReason.STALE_SOURCE,
            )
        if worker.lifecycle_state in _INELIGIBLE_WORKER_LIFECYCLE_STATES:
            return self._rejected(
                correlation=base_correlation,
                reason=WorkerExecutionDispatchRejectionReason.WORKER_NOT_ELIGIBLE,
            )

        if request.goal_id is not None:
            rejection = self._validate_goal_driven_source(request=request)
            if rejection is not None:
                return self._rejected(
                    correlation=base_correlation,
                    reason=rejection,
                )

        try:
            authority_context = self._admission_service.prepare(
                WorkerExecutionAuthorityRequest(
                    worker_instance_id=request.worker_instance_id,
                    requested_authority_scopes=request.requested_scopes,
                )
            )
        except WorkerExecutionAuthorityDenied:
            return self._rejected(
                correlation=base_correlation,
                reason=WorkerExecutionDispatchRejectionReason.COLLABORATIVE_AUTHORITY_DENIED,
            )

        principal = authority_context.resolved_principal
        root_admission = self._root_authority_admission.authorize(
            RootExecutionAuthorityAdmissionRequest(
                tenant_id=principal.tenant_id,
                workspace_id=principal.workspace_id,
                principal_id=principal.principal_id,
                collaborative_authority_scopes=authority_context.collaborative_authority_scopes,
                effective_authority_decision=authority_context.effective_authority_decision,
            )
        )
        if root_admission.disposition is not RootExecutionAuthorityAdmissionDisposition.ALLOWED:
            if root_admission.disposition is RootExecutionAuthorityAdmissionDisposition.UNAVAILABLE:
                return self._unavailable(correlation=base_correlation)
            return self._rejected(
                correlation=base_correlation,
                reason=WorkerExecutionDispatchRejectionReason.RUNTIME_AUTHORITY_DENIED,
            )
        assert root_admission.trusted_parent_execution_authority is not None

        try:
            intake_result = await self._execution_intake.dispatch(
                CanonicalExecutionIntakeRequest(
                    payload=request.runtime_request,
                    trusted_parent_execution_authority=root_admission.trusted_parent_execution_authority,
                    tenant_id=principal.tenant_id,
                    run_id=request.run_id,
                    attempt_id=request.attempt_id,
                )
            )
        except CanonicalExecutionInvocationFailed as exc:
            return WorkerExecutionDispatchResult(
                disposition=WorkerExecutionDispatchDisposition.FAILED,
                correlation=WorkerExecutionCorrelation(
                    worker_instance_id=request.worker_instance_id,
                    source=request.source,
                    run_id=exc.run_id,
                    attempt_id=exc.attempt_id,
                    execution_id=exc.execution_id,
                    goal_id=request.goal_id,
                    responsibility_id=request.responsibility_id,
                    wake_up_id=request.wake_up_id,
                    collaborative_work_ref=request.collaborative_work_ref,
                    created_at=base_correlation.created_at,
                ),
                failure_reason=str(exc.cause or exc),
            )

        return WorkerExecutionDispatchResult(
            disposition=WorkerExecutionDispatchDisposition.DISPATCHED,
            correlation=WorkerExecutionCorrelation(
                worker_instance_id=request.worker_instance_id,
                source=request.source,
                run_id=intake_result.run_id,
                attempt_id=intake_result.attempt_id,
                execution_id=intake_result.execution_id,
                goal_id=request.goal_id,
                responsibility_id=request.responsibility_id,
                wake_up_id=request.wake_up_id,
                collaborative_work_ref=request.collaborative_work_ref,
                created_at=base_correlation.created_at,
            ),
            runtime_result=intake_result.result,
        )

    def _validate_goal_driven_source(
        self,
        *,
        request: WorkerExecutionDispatchRequest[InputT, OutputT],
    ) -> WorkerExecutionDispatchRejectionReason | None:
        assert request.goal_id is not None
        assert request.goal_revision is not None
        goal = self._worker_goal_repository.get(goal_id=request.goal_id)
        if goal is None:
            return WorkerExecutionDispatchRejectionReason.OWNERSHIP_MISMATCH
        if goal.status is not WorkerGoalStatus.ACTIVE:
            return WorkerExecutionDispatchRejectionReason.STALE_SOURCE
        if goal.revision != request.goal_revision:
            return WorkerExecutionDispatchRejectionReason.STALE_SOURCE
        if request.responsibility_id is not None:
            responsibility = self._responsibility_repository.get(
                responsibility_id=request.responsibility_id,
            )
            if responsibility is None:
                return WorkerExecutionDispatchRejectionReason.OWNERSHIP_MISMATCH
            if responsibility.worker_instance_id != request.worker_instance_id:
                return WorkerExecutionDispatchRejectionReason.OWNERSHIP_MISMATCH
            if responsibility.status is not ResponsibilityStatus.ACTIVE:
                return WorkerExecutionDispatchRejectionReason.STALE_SOURCE
            if goal.responsibility_id != request.responsibility_id:
                return WorkerExecutionDispatchRejectionReason.OWNERSHIP_MISMATCH
        return None

    @staticmethod
    def _rejected(
        *,
        correlation: WorkerExecutionCorrelation,
        reason: WorkerExecutionDispatchRejectionReason,
    ) -> WorkerExecutionDispatchResult[OutputT]:
        return WorkerExecutionDispatchResult(
            disposition=WorkerExecutionDispatchDisposition.REJECTED,
            correlation=correlation,
            rejection_reason=reason,
        )

    @staticmethod
    def _unavailable(
        *,
        correlation: WorkerExecutionCorrelation,
    ) -> WorkerExecutionDispatchResult[OutputT]:
        return WorkerExecutionDispatchResult(
            disposition=WorkerExecutionDispatchDisposition.UNAVAILABLE,
            correlation=correlation,
            rejection_reason=WorkerExecutionDispatchRejectionReason.RUNTIME_UNAVAILABLE,
        )
