# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Audit port availability check (SELF-HEALING R6.4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.evaluation_audit import AutonomyEvaluationAuditRecorder
from intergrax.contracts.self_healing.autonomy.execution_audit import AutonomyExecutionAuditRepository
from intergrax.contracts.self_healing.autonomy.qualification.context import AutonomySafetyQualificationContext
from intergrax.contracts.self_healing.autonomy.qualification.result import (
    AutonomyQualificationStatus,
    AutonomySafetyCheckResult,
    AutonomySafetyIssue,
)

_CHECK_ID = "audit_capability"


@dataclass(frozen=True, slots=True)
class AuditCapabilityCheck:
    @property
    def check_id(self) -> str:
        return _CHECK_ID

    def run(self, context: AutonomySafetyQualificationContext) -> AutonomySafetyCheckResult:
        issues: list[AutonomySafetyIssue] = []
        recorder = context.evaluation_audit_recorder
        exec_repo = context.execution_audit_repository
        if recorder is None:
            issues.append(
                AutonomySafetyIssue(
                    code="missing_evaluation_audit_recorder",
                    message="AutonomyEvaluationAuditRecorder port is required for R6.2 audit replay.",
                    check_id=_CHECK_ID,
                    severity=AutonomyQualificationStatus.FAILED,
                )
            )
        elif not isinstance(recorder, AutonomyEvaluationAuditRecorder):
            issues.append(
                AutonomySafetyIssue(
                    code="invalid_evaluation_audit_recorder",
                    message="evaluation_audit_recorder does not satisfy AutonomyEvaluationAuditRecorder.",
                    check_id=_CHECK_ID,
                    severity=AutonomyQualificationStatus.FAILED,
                )
            )
        if exec_repo is None:
            issues.append(
                AutonomySafetyIssue(
                    code="missing_execution_audit_repository",
                    message="AutonomyExecutionAuditRepository port is required for R6.3 guard audit.",
                    check_id=_CHECK_ID,
                    severity=AutonomyQualificationStatus.FAILED,
                )
            )
        elif not isinstance(exec_repo, AutonomyExecutionAuditRepository):
            issues.append(
                AutonomySafetyIssue(
                    code="invalid_execution_audit_repository",
                    message="execution_audit_repository does not satisfy AutonomyExecutionAuditRepository.",
                    check_id=_CHECK_ID,
                    severity=AutonomyQualificationStatus.FAILED,
                )
            )
        status = AutonomyQualificationStatus.FAILED if issues else AutonomyQualificationStatus.PASS
        return AutonomySafetyCheckResult(check_id=_CHECK_ID, status=status, issues=tuple(issues))


__all__ = ["AuditCapabilityCheck"]
