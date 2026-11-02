# © Artur Czarnecki. All rights reserved.

"""Audit mode policies and exit-code aggregation."""

from __future__ import annotations

from intergrax.platform.enterprise_audit.contracts import (
    AggregateAuditStatus,
    AuditFinding,
    AuditMode,
    AuditSeverity,
    CheckRunOutcome,
    EnterpriseAuditExitCode,
)


def aggregate_status(
    findings: tuple[AuditFinding, ...],
    *,
    incomplete: bool,
    internal_error: bool,
) -> AggregateAuditStatus:
    if internal_error:
        return AggregateAuditStatus.INTERNAL_FAILURE
    if incomplete:
        return AggregateAuditStatus.INCOMPLETE if not findings else AggregateAuditStatus.PARTIAL
    severities = {f.severity for f in findings}
    if AuditSeverity.BLOCKER in severities or AuditSeverity.ERROR in severities:
        return AggregateAuditStatus.BLOCKED
    if AuditSeverity.ENVIRONMENT in severities:
        return AggregateAuditStatus.ENVIRONMENT_FAILURE
    if findings:
        return AggregateAuditStatus.PARTIAL
    return AggregateAuditStatus.PASS


def compute_exit_code(
    status: AggregateAuditStatus,
    outcomes: tuple[CheckRunOutcome, ...],
) -> EnterpriseAuditExitCode:
    if status is AggregateAuditStatus.INTERNAL_FAILURE:
        return EnterpriseAuditExitCode.INTERNAL
    if any(o.exit_code == 3 for o in outcomes if o.executed):
        return EnterpriseAuditExitCode.INTERNAL
    if status in {AggregateAuditStatus.BLOCKED, AggregateAuditStatus.ENVIRONMENT_FAILURE}:
        return EnterpriseAuditExitCode.VIOLATION
    if status in {AggregateAuditStatus.PARTIAL, AggregateAuditStatus.INCOMPLETE}:
        return EnterpriseAuditExitCode.INCOMPLETE
    return EnterpriseAuditExitCode.CLEAN


def fast_floor_incomplete(
    mode: AuditMode,
    all_checks: tuple,
    outcomes: tuple[CheckRunOutcome, ...],
) -> bool:
    if mode is not AuditMode.FAST:
        return False
    mandatory_ids = {c.check_id for c in all_checks if getattr(c, "fast_mandatory", False)}
    executed_ids = {o.check_id for o in outcomes if o.executed}
    missing = mandatory_ids - executed_ids
    if missing:
        return True
    for outcome in outcomes:
        if outcome.check_id in mandatory_ids and outcome.findings:
            if any(f.severity is AuditSeverity.TRACKED_DEBT for f in outcome.findings):
                return True
    return False


def default_limitations(mode: AuditMode) -> tuple[str, ...]:
    base = (
        "Path B: FRZ-PRD live qualification dimension remains OPEN — see QUALIFICATION_LIMITATIONS.",
        "Historical gate PASS at scoped SHA does not imply current-HEAD PASS (ECAP-P2).",
    )
    if mode is AuditMode.FAST:
        return base + ("FAST mode skips internal_pending and non-executable partial entrypoints.",)
    return base
