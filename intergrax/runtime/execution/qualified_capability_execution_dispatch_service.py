# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ingress dedup and canonical root launch for qualified capabilities (UCA-6C-R2/R3)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
    QualifiedCapabilityExecutionDispatchPort,
    QualifiedCapabilityExecutionDispatchRequest,
    QualifiedCapabilityExecutionDispatchResult,
)
from intergrax.contracts.execution.qualified_capability_execution_intake import (
    QualifiedCapabilityExecutionDelegateResult,
    QualifiedCapabilityExecutionIntakePayload,
)
from intergrax.contracts.root_execution_launch import (
    RootExecutionLaunchDisposition,
    RootExecutionLaunchPort,
    RootExecutionLaunchRequest,
)
from intergrax.contracts.root_execution_operation import RootExecutionOperation
from intergrax.runtime.execution.qualified_capability_execution_runtime_delegate import (
    QualifiedCapabilityExecutionRuntimeDelegate,
)
from intergrax.tools._shared.async_dispatch import run_async


@dataclass(frozen=True, slots=True)
class _IngressLedgerEntry:
    result: QualifiedCapabilityExecutionDispatchResult


class QualifiedCapabilityExecutionDispatchService(
    QualifiedCapabilityExecutionDispatchPort,
):
    """Process-local ingress dedup — canonical execution authority is ExecutionRuntime."""

    def __init__(
        self,
        *,
        root_execution_launcher: RootExecutionLaunchPort[
            QualifiedCapabilityExecutionIntakePayload,
            QualifiedCapabilityExecutionDelegateResult,
        ],
        runtime_delegate: QualifiedCapabilityExecutionRuntimeDelegate,
    ) -> None:
        self._launcher = root_execution_launcher
        self._runtime_delegate = runtime_delegate
        self._ledger: dict[tuple[str, str], _IngressLedgerEntry] = {}
        self._lock = threading.RLock()

    @property
    def dispatch_side_effects(self) -> int:
        return self._runtime_delegate.execute_calls

    def dispatch(
        self,
        request: QualifiedCapabilityExecutionDispatchRequest,
    ) -> QualifiedCapabilityExecutionDispatchResult:
        ledger_key = (request.tenant_id, request.execution_request_id)
        with self._lock:
            existing = self._ledger.get(ledger_key)
            if existing is not None:
                return existing.result

            if request.governance_approval_evidence is not None:
                evidence = request.governance_approval_evidence
                if evidence.tenant_id != request.tenant_id:
                    return QualifiedCapabilityExecutionDispatchResult(
                        disposition=QualifiedCapabilityExecutionDispatchDisposition.REJECTED,
                        execution_request_id=request.execution_request_id,
                        reason_detail="governance_approval_evidence_tenant_mismatch",
                    )

            payload = QualifiedCapabilityExecutionIntakePayload(
                execution_request_id=request.execution_request_id,
                execution_target=request.execution_target,
                tenant_id=request.tenant_id,
                task_id=request.task_id,
                worker_instance_id=request.worker_instance_id,
                worker_need_id=request.worker_need_id,
                resume_operation_id=request.resume_operation_id,
                binding_operation_id=request.binding_operation_id,
                qualification_request_id=request.qualification_request_id,
                acquisition_request_id=request.acquisition_request_id,
                qualified_subject_reference=request.qualified_subject_reference,
                requested_at=request.requested_at,
                admitted_governance_identity=request.admitted_governance_identity,
                effective_authority_decision=request.effective_authority_decision,
                collaborative_authority_scopes=request.collaborative_authority_scopes,
                governance_approval_evidence=request.governance_approval_evidence,
            )
            launch_result = run_async(
                self._launcher.launch(
                    RootExecutionLaunchRequest(
                        admitted_governance_identity=request.admitted_governance_identity,
                        root_execution_operation=RootExecutionOperation.ROOT_WORKER_DISPATCH,
                        collaborative_authority_scopes=request.collaborative_authority_scopes,
                        effective_authority_decision=request.effective_authority_decision,
                        payload=payload,
                        run_id=request.run_id,
                        attempt_id=request.attempt_id,
                        task_id=request.task_id,
                    ),
                ),
            )
            result = _map_launch_result(request, launch_result)
            self._ledger[ledger_key] = _IngressLedgerEntry(result=result)
            return result


def _map_launch_result(
    request: QualifiedCapabilityExecutionDispatchRequest,
    launch_result,
) -> QualifiedCapabilityExecutionDispatchResult:
    if launch_result.disposition is RootExecutionLaunchDisposition.UNAVAILABLE:
        return QualifiedCapabilityExecutionDispatchResult(
            disposition=QualifiedCapabilityExecutionDispatchDisposition.UNAVAILABLE,
            execution_request_id=request.execution_request_id,
            reason_detail="canonical_execution_unavailable",
        )
    if launch_result.disposition is not RootExecutionLaunchDisposition.LAUNCHED:
        return QualifiedCapabilityExecutionDispatchResult(
            disposition=QualifiedCapabilityExecutionDispatchDisposition.REJECTED,
            execution_request_id=request.execution_request_id,
            reason_detail="canonical_execution_admission_denied",
        )
    assert launch_result.intake_result is not None
    intake = launch_result.intake_result
    delegate_result = intake.result
    if (
        delegate_result.disposition
        is QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED
    ):
        return QualifiedCapabilityExecutionDispatchResult(
            disposition=QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED,
            execution_request_id=request.execution_request_id,
            run_id=intake.run_id,
            attempt_id=intake.attempt_id,
            execution_id=intake.execution_id,
        )
    return QualifiedCapabilityExecutionDispatchResult(
        disposition=delegate_result.disposition,
        execution_request_id=request.execution_request_id,
        run_id=intake.run_id,
        attempt_id=intake.attempt_id,
        execution_id=intake.execution_id,
        reason_detail=delegate_result.reason_detail,
    )


__all__ = ["QualifiedCapabilityExecutionDispatchService"]
