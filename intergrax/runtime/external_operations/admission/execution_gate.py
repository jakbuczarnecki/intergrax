# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Admission gate before provider execution (R1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from intergrax.contracts.external_operations.admission import (
    ExternalOperationAdmission,
    ExternalOperationAdmissionContext,
    OperationAdmissionDecision,
    OperationAdmissionVerdict,
)
from intergrax.contracts.external_operations.attempt import (
    ExternalOperationAttempt,
    ExternalOperationAttemptLifecycle,
    mint_external_operation_attempt_id,
)
from intergrax.contracts.external_operations.evidence import (
    ExternalOperationEvidence,
    ExternalOperationEvidenceKind,
)
from intergrax.contracts.external_operations.intent import ExternalOperationIntent
from intergrax.contracts.external_operations.safety import (
    ExternalOperationAdmissionDeniedError,
    ExternalOperationApprovalRequiredError,
    ExternalOperationExecutionForbiddenError,
)
from intergrax.runtime.external_operations.admission.audit_chain import (
    ExternalOperationAuditChain,
)


@dataclass
class ExternalOperationExecutionGate:
    admission: ExternalOperationAdmission
    audit_chain: ExternalOperationAuditChain | None = None
    actor: str = "runtime.external_operation_gate"
    _decisions_by_attempt: dict[str, OperationAdmissionDecision] | None = None

    def __post_init__(self) -> None:
        if self._decisions_by_attempt is None:
            self._decisions_by_attempt = {}

    def admit_intent(
        self,
        intent: ExternalOperationIntent,
        *,
        context: ExternalOperationAdmissionContext,
        provider_id: str | None = None,
    ) -> ExternalOperationAttempt:
        decision = self.admission.evaluate(intent, context)
        attempt = ExternalOperationAttempt(
            operation_attempt_id=mint_external_operation_attempt_id(),
            intent=intent,
            tenant_id=intent.tenant_id,
            task_id=intent.task_id,
        )
        if decision.verdict is OperationAdmissionVerdict.DENY:
            if self.audit_chain is not None:
                self.audit_chain.append(
                    attempt=attempt,
                    admission=decision,
                    actor=self.actor,
                )
            raise ExternalOperationAdmissionDeniedError(decision.reason)
        if decision.verdict is OperationAdmissionVerdict.REQUIRES_APPROVAL:
            if not (
                context.human_approval_granted and context.approval_id is not None
            ):
                if self.audit_chain is not None:
                    self.audit_chain.append(
                        attempt=attempt,
                        admission=decision,
                        actor=self.actor,
                    )
                raise ExternalOperationApprovalRequiredError(decision.reason)
        admitted = attempt.transition(ExternalOperationAttemptLifecycle.ADMITTED)
        if provider_id is not None:
            admitted = admitted.bind_provider(provider_id)
        self._decisions_by_attempt[admitted.operation_attempt_id] = decision
        self._append_audit(admitted, decision)
        return admitted

    def _append_audit(
        self,
        attempt: ExternalOperationAttempt,
        decision: OperationAdmissionDecision | None,
    ) -> None:
        if self.audit_chain is None or decision is None:
            return
        self.audit_chain.append(
            attempt=attempt,
            admission=decision,
            actor=self.actor,
        )

    def begin_execution(
        self,
        attempt: ExternalOperationAttempt,
        *,
        admission: OperationAdmissionDecision | None = None,
    ) -> ExternalOperationAttempt:
        if attempt.lifecycle is not ExternalOperationAttemptLifecycle.ADMITTED:
            raise ExternalOperationExecutionForbiddenError(
                "execution requires ADMITTED lifecycle"
            )
        executing = attempt.transition(ExternalOperationAttemptLifecycle.EXECUTING)
        decision = admission or self._decisions_by_attempt.get(attempt.operation_attempt_id)
        self._append_audit(executing, decision)
        return executing

    def complete_success(
        self,
        attempt: ExternalOperationAttempt,
        *,
        admission: OperationAdmissionDecision | None = None,
    ) -> ExternalOperationAttempt:
        if attempt.lifecycle is not ExternalOperationAttemptLifecycle.EXECUTING:
            raise ExternalOperationExecutionForbiddenError(
                "terminal success requires EXECUTING lifecycle"
            )
        terminal = attempt.transition(ExternalOperationAttemptLifecycle.SUCCEEDED)
        decision = admission or self._decisions_by_attempt.get(attempt.operation_attempt_id)
        self._append_audit(terminal, decision)
        return terminal

    def complete_failure(
        self,
        attempt: ExternalOperationAttempt,
        *,
        admission: OperationAdmissionDecision | None = None,
    ) -> ExternalOperationAttempt:
        if attempt.lifecycle is not ExternalOperationAttemptLifecycle.EXECUTING:
            raise ExternalOperationExecutionForbiddenError(
                "terminal failure requires EXECUTING lifecycle"
            )
        terminal = attempt.transition(ExternalOperationAttemptLifecycle.FAILED)
        decision = admission or self._decisions_by_attempt.get(attempt.operation_attempt_id)
        self._append_audit(terminal, decision)
        return terminal

    @staticmethod
    def build_denial_evidence(
        attempt: ExternalOperationAttempt,
        *,
        reason: str,
    ) -> ExternalOperationEvidence:
        return ExternalOperationEvidence(
            evidence_id=f"ext_op_ev_{uuid4().hex}",
            attempt_id=attempt.operation_attempt_id,
            intent_id=attempt.intent.intent_id,
            tenant_id=attempt.intent.tenant_id,
            kind=ExternalOperationEvidenceKind.ADMISSION_DENIAL,
            safe_summary=reason[:512],
            recorded_at=datetime.now(timezone.utc),
        )
