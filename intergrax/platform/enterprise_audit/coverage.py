# © Artur Czarnecki. All rights reserved.

"""Honest coverage_label computation."""

from __future__ import annotations

from intergrax.platform.enterprise_audit.contracts import (
    AuditFinding,
    AuditMode,
    AuditSeverity,
    CheckRunOutcome,
)


def compute_coverage_label(
    mode: AuditMode,
    *,
    total_checks: int,
    planned: int,
    outcomes: tuple[CheckRunOutcome, ...],
    findings: tuple[AuditFinding, ...],
    certifying_run: bool = True,
    filtered: bool = False,
) -> str:
    if certifying_run and filtered and mode is not AuditMode.CHANGED_SINCE:
        return "PARTIAL"
    executed = sum(1 for o in outcomes if o.executed)
    skipped = sum(1 for o in outcomes if o.skipped)
    blocking = any(f.severity in {AuditSeverity.BLOCKER, AuditSeverity.ERROR} for f in findings)
    if mode is AuditMode.FAST:
        return "PARTIAL"
    if mode in {AuditMode.FULL, AuditMode.FREEZE}:
        if executed == total_checks and skipped == 0 and planned == total_checks:
            if not blocking:
                return "FULL"
        return "PARTIAL"
    if mode is AuditMode.CHANGED_SINCE:
        return "PARTIAL" if executed > 0 else "NONE"
    return "PARTIAL"
