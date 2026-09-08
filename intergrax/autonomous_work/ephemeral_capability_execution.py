# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded A1 ephemeral capability execution service (AW-7B).

Validates AW-7A EPHEMERAL_GENERATION_CANDIDATE decisions and correlation,
then delegates to a provider-neutral execution port. Does not generate code,
resolve sandbox, approve HITL, or mutate registries.
"""

from __future__ import annotations

from datetime import datetime

from intergrax.autonomous_work.ephemeral_capability_execution_ports import (
    WorkerEphemeralCapabilityExecutionPort,
)
from intergrax.contracts.autonomous_work.ephemeral_capability_execution import (
    EPHEMERAL_EXECUTION_POLICY_VERSION,
    WorkerEphemeralCapabilityExecutionReasonCode,
    WorkerEphemeralCapabilityExecutionRequest,
    WorkerEphemeralCapabilityExecutionResult,
    WorkerEphemeralCapabilityExecutionStatus,
    validate_a1_ephemeral_execution_eligibility,
)


class WorkerEphemeralCapabilityExecutionService:
    """Stateless A1 execution orchestration — lifecycle owned by provider adapter."""

    def __init__(self, *, execution_port: WorkerEphemeralCapabilityExecutionPort) -> None:
        self._execution_port = execution_port

    def execute(
        self,
        request: WorkerEphemeralCapabilityExecutionRequest,
        *,
        executed_at: datetime | None = None,
    ) -> WorkerEphemeralCapabilityExecutionResult:
        timestamp = executed_at or request.requested_at
        rejection = validate_a1_ephemeral_execution_eligibility(request)
        if rejection is not None:
            return _eligibility_rejection(request, reason_code=rejection, executed_at=timestamp)
        return self._execution_port.execute(request)


def _eligibility_rejection(
    request: WorkerEphemeralCapabilityExecutionRequest,
    *,
    reason_code: WorkerEphemeralCapabilityExecutionReasonCode,
    executed_at: datetime,
) -> WorkerEphemeralCapabilityExecutionResult:
    status = (
        WorkerEphemeralCapabilityExecutionStatus.CONFLICT
        if reason_code
        in {
            WorkerEphemeralCapabilityExecutionReasonCode.CORRELATION_CONFLICT,
            WorkerEphemeralCapabilityExecutionReasonCode.CANDIDATE_DECISION_MISMATCH,
        }
        else WorkerEphemeralCapabilityExecutionStatus.DENIED
    )
    return WorkerEphemeralCapabilityExecutionResult(
        status=status,
        reason_code=reason_code,
        worker_instance_id=request.worker_instance_id,
        acquisition_decision_id=request.acquisition_decision.decision_id,
        need_id=request.need_id,
        evidence_refs=request.evidence_refs,
        executed_at=executed_at,
        execution_policy_version=EPHEMERAL_EXECUTION_POLICY_VERSION,
        error_code=reason_code.value,
    )
