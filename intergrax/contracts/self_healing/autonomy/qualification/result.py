# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Qualification outcome models (SELF-HEALING R6.4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


AUTONOMY_QUALIFICATION_CONTRACT_VERSION = "1.0.0"


class AutonomyQualificationStatus(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class AutonomySafetyIssue:
    code: str
    message: str
    check_id: str
    severity: AutonomyQualificationStatus

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("code required")
        if not self.message.strip():
            raise ValueError("message required")
        if not self.check_id.strip():
            raise ValueError("check_id required")
        if self.severity is AutonomyQualificationStatus.PASS:
            raise ValueError("issue severity cannot be PASS")


@dataclass(frozen=True, slots=True)
class AutonomySafetyCheckResult:
    check_id: str
    status: AutonomyQualificationStatus
    issues: tuple[AutonomySafetyIssue, ...] = ()

    def __post_init__(self) -> None:
        if not self.check_id.strip():
            raise ValueError("check_id required")


@dataclass(frozen=True, slots=True)
class AutonomyQualificationAuditInfo:
    validation_id: str
    qualified_at: datetime
    checks_executed: tuple[str, ...]
    contract_version: str

    def __post_init__(self) -> None:
        if not self.validation_id.strip():
            raise ValueError("validation_id required")
        if not self.contract_version.strip():
            raise ValueError("contract_version required")
        if not self.checks_executed:
            raise ValueError("checks_executed required")


@dataclass(frozen=True, slots=True)
class AutonomyQualificationResult:
    validation_id: str
    status: AutonomyQualificationStatus
    issues: tuple[AutonomySafetyIssue, ...]
    check_results: tuple[AutonomySafetyCheckResult, ...]
    audit: AutonomyQualificationAuditInfo

    def __post_init__(self) -> None:
        if not self.validation_id.strip():
            raise ValueError("validation_id required")
        if self.audit.validation_id != self.validation_id:
            raise ValueError("audit.validation_id mismatch")


__all__ = [
    "AUTONOMY_QUALIFICATION_CONTRACT_VERSION",
    "AutonomyQualificationAuditInfo",
    "AutonomyQualificationResult",
    "AutonomyQualificationStatus",
    "AutonomySafetyCheckResult",
    "AutonomySafetyIssue",
]
