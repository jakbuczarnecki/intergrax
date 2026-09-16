# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderCheckResult,
    MemoryProviderCheckSeverity,
    MemoryProviderQualificationFailureReason,
)


def passed(
    *,
    check_id: str,
    capability: MemoryProviderCapabilityKind,
    severity: MemoryProviderCheckSeverity,
) -> MemoryProviderCheckResult:
    return MemoryProviderCheckResult(
        check_id=check_id,
        capability=capability,
        severity=severity,
        passed=True,
    )


def failed(
    *,
    check_id: str,
    capability: MemoryProviderCapabilityKind,
    severity: MemoryProviderCheckSeverity,
    reason_code: MemoryProviderQualificationFailureReason,
    detail: str | None = None,
) -> MemoryProviderCheckResult:
    return MemoryProviderCheckResult(
        check_id=check_id,
        capability=capability,
        severity=severity,
        passed=False,
        reason_code=reason_code,
        detail=detail,
    )
