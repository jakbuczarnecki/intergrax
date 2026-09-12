# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Execution boundary presence check (SELF-HEALING R6.4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.execution_boundary import AutonomyExecutionBoundary
from intergrax.contracts.self_healing.autonomy.guard import AutonomyExecutionGuard
from intergrax.contracts.self_healing.autonomy.qualification.context import AutonomySafetyQualificationContext
from intergrax.contracts.self_healing.autonomy.qualification.result import (
    AutonomyQualificationStatus,
    AutonomySafetyCheckResult,
    AutonomySafetyIssue,
)

_CHECK_ID = "execution_boundary_safety"


@dataclass(frozen=True, slots=True)
class ExecutionBoundarySafetyCheck:
    @property
    def check_id(self) -> str:
        return _CHECK_ID

    def run(self, context: AutonomySafetyQualificationContext) -> AutonomySafetyCheckResult:
        issues: list[AutonomySafetyIssue] = []
        guard = context.execution_guard
        boundary = context.execution_boundary
        if guard is None:
            issues.append(
                AutonomySafetyIssue(
                    code="missing_execution_guard",
                    message="AutonomyExecutionGuard is required before controlled execution.",
                    check_id=_CHECK_ID,
                    severity=AutonomyQualificationStatus.FAILED,
                )
            )
        elif not isinstance(guard, AutonomyExecutionGuard):
            issues.append(
                AutonomySafetyIssue(
                    code="invalid_execution_guard",
                    message="execution_guard does not satisfy AutonomyExecutionGuard.",
                    check_id=_CHECK_ID,
                    severity=AutonomyQualificationStatus.FAILED,
                )
            )
        if boundary is None:
            issues.append(
                AutonomySafetyIssue(
                    code="missing_execution_boundary",
                    message="AutonomyExecutionBoundary is required for spine integration.",
                    check_id=_CHECK_ID,
                    severity=AutonomyQualificationStatus.FAILED,
                )
            )
        elif not isinstance(boundary, AutonomyExecutionBoundary):
            issues.append(
                AutonomySafetyIssue(
                    code="invalid_execution_boundary",
                    message="execution_boundary does not satisfy AutonomyExecutionBoundary.",
                    check_id=_CHECK_ID,
                    severity=AutonomyQualificationStatus.FAILED,
                )
            )
        if context.direct_executor_access_enabled:
            issues.append(
                AutonomySafetyIssue(
                    code="direct_executor_access",
                    message="Direct executor access bypasses autonomy guard and boundary.",
                    check_id=_CHECK_ID,
                    severity=AutonomyQualificationStatus.FAILED,
                )
            )
        status = _status_from_issues(issues)
        return AutonomySafetyCheckResult(check_id=_CHECK_ID, status=status, issues=tuple(issues))


def _status_from_issues(issues: list[AutonomySafetyIssue]) -> AutonomyQualificationStatus:
    if any(issue.severity is AutonomyQualificationStatus.FAILED for issue in issues):
        return AutonomyQualificationStatus.FAILED
    if any(issue.severity is AutonomyQualificationStatus.WARNING for issue in issues):
        return AutonomyQualificationStatus.WARNING
    return AutonomyQualificationStatus.PASS


__all__ = ["ExecutionBoundarySafetyCheck"]
