# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy qualification orchestration — inspect only (SELF-HEALING R6.4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from intergrax.contracts.self_healing.autonomy.ids import mint_autonomy_qualification_validation_id
from intergrax.contracts.self_healing.autonomy.qualification.context import AutonomySafetyQualificationContext
from intergrax.contracts.self_healing.autonomy.qualification.repository import AutonomyQualificationRepository
from intergrax.contracts.self_healing.autonomy.qualification.result import (
    AUTONOMY_QUALIFICATION_CONTRACT_VERSION,
    AutonomyQualificationAuditInfo,
    AutonomyQualificationResult,
    AutonomyQualificationStatus,
    AutonomySafetyCheckResult,
    AutonomySafetyIssue,
)
from intergrax.contracts.self_healing.autonomy.qualification.safety_check import AutonomySafetyCheck


def aggregate_qualification_status(
    check_results: tuple[AutonomySafetyCheckResult, ...],
) -> AutonomyQualificationStatus:
    if any(result.status is AutonomyQualificationStatus.FAILED for result in check_results):
        return AutonomyQualificationStatus.FAILED
    if any(result.status is AutonomyQualificationStatus.WARNING for result in check_results):
        return AutonomyQualificationStatus.WARNING
    return AutonomyQualificationStatus.PASS


@dataclass(frozen=True, slots=True)
class AutonomyQualificationService:
    safety_checks: tuple[AutonomySafetyCheck, ...]
    repository: AutonomyQualificationRepository | None = None
    _validator_id: str = "platform.autonomy_qualification"

    @property
    def validator_id(self) -> str:
        return self._validator_id

    def qualify(self, context: AutonomySafetyQualificationContext) -> AutonomyQualificationResult:
        validation_id = mint_autonomy_qualification_validation_id()
        qualified_at = datetime.now(tz=timezone.utc)
        check_results: list[AutonomySafetyCheckResult] = []
        for check in self.safety_checks:
            check_results.append(check.run(context))
        aggregated_issues: list[AutonomySafetyIssue] = []
        for result in check_results:
            aggregated_issues.extend(result.issues)
        checks_executed = tuple(result.check_id for result in check_results)
        audit = AutonomyQualificationAuditInfo(
            validation_id=validation_id,
            qualified_at=qualified_at,
            checks_executed=checks_executed,
            contract_version=AUTONOMY_QUALIFICATION_CONTRACT_VERSION,
        )
        outcome = AutonomyQualificationResult(
            validation_id=validation_id,
            status=aggregate_qualification_status(tuple(check_results)),
            issues=tuple(aggregated_issues),
            check_results=tuple(check_results),
            audit=audit,
        )
        if self.repository is not None:
            self.repository.save(outcome)
        return outcome


__all__ = [
    "AutonomyQualificationService",
    "aggregate_qualification_status",
]
