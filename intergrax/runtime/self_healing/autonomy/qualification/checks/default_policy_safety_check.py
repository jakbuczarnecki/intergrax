# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Default autonomy level posture check (SELF-HEALING R6.4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.qualification.context import AutonomySafetyQualificationContext
from intergrax.contracts.self_healing.autonomy.qualification.result import (
    AutonomyQualificationStatus,
    AutonomySafetyCheckResult,
    AutonomySafetyIssue,
)

_CHECK_ID = "default_policy_safety"

_SAFE_DEFAULT_LEVELS: frozenset[AutonomyLevel] = frozenset(
    {
        AutonomyLevel.OBSERVE_ONLY,
        AutonomyLevel.RECOMMEND_ONLY,
        AutonomyLevel.APPROVAL_REQUIRED,
    }
)


@dataclass(frozen=True, slots=True)
class DefaultPolicySafetyCheck:
    @property
    def check_id(self) -> str:
        return _CHECK_ID

    def run(self, context: AutonomySafetyQualificationContext) -> AutonomySafetyCheckResult:
        level = context.default_autonomy_level
        issues: list[AutonomySafetyIssue] = []
        if level is AutonomyLevel.FULL_AUTONOMY:
            issues.append(
                AutonomySafetyIssue(
                    code="forbidden_full_autonomy_default",
                    message="FULL_AUTONOMY cannot be used as a default autonomy level.",
                    check_id=_CHECK_ID,
                    severity=AutonomyQualificationStatus.FAILED,
                )
            )
        elif level is AutonomyLevel.CONTROLLED_EXECUTION:
            issues.append(
                AutonomySafetyIssue(
                    code="unsafe_controlled_execution_default",
                    message="CONTROLLED_EXECUTION must not be the platform default posture.",
                    check_id=_CHECK_ID,
                    severity=AutonomyQualificationStatus.FAILED,
                )
            )
        elif level not in _SAFE_DEFAULT_LEVELS:
            issues.append(
                AutonomySafetyIssue(
                    code="unknown_autonomy_default",
                    message=f"Unrecognized default autonomy level: {level.value}.",
                    check_id=_CHECK_ID,
                    severity=AutonomyQualificationStatus.FAILED,
                )
            )
        status = AutonomyQualificationStatus.FAILED if issues else AutonomyQualificationStatus.PASS
        return AutonomySafetyCheckResult(check_id=_CHECK_ID, status=status, issues=tuple(issues))


__all__ = ["DefaultPolicySafetyCheck"]
