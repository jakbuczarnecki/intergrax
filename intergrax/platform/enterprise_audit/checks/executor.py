# © Artur Czarnecki. All rights reserved.

"""Per-check execution and finding materialization."""

from __future__ import annotations

from intergrax.platform.enterprise_audit.contracts import (
    AuditCategory,
    AuditFinding,
    AuditMode,
    AuditSeverity,
    CheckRunOutcome,
    ManifestCheck,
)
from intergrax.platform.enterprise_audit.sandbox import CheckExecutionSandbox, SandboxResult


def _finding_from_failure(check: ManifestCheck, message: str) -> AuditFinding:
    return AuditFinding(
        finding_id=f"{check.check_id}:FAIL",
        check_id=check.check_id,
        severity=check.severity_default,
        category=check.category,
        message=message,
        source_path=check.entrypoint,
    )


def skip_prerequisite(check: ManifestCheck) -> CheckRunOutcome:
    finding = AuditFinding(
        finding_id=f"{check.check_id}:SKIPPED_PREREQ",
        check_id=check.check_id,
        severity=AuditSeverity.ERROR,
        category=AuditCategory.INTERNAL,
        message="Skipped due to failed prerequisite",
        source_path=check.entrypoint,
    )
    return CheckRunOutcome(
        check_id=check.check_id,
        executed=False,
        skipped=True,
        skip_reason="skipped_prerequisite",
        findings=(finding,),
    )


def execute_manifest_check(
    check: ManifestCheck,
    sandbox: CheckExecutionSandbox,
    *,
    mode: AuditMode,
) -> CheckRunOutcome:
    if check.entrypoint_kind == "internal_pending":
        severity = AuditSeverity.TRACKED_DEBT if mode is AuditMode.FAST else AuditSeverity.ERROR
        finding = AuditFinding(
            finding_id=f"{check.check_id}:PENDING",
            check_id=check.check_id,
            severity=severity,
            category=AuditCategory.INTERNAL,
            message=f"Check adapter pending implementation: {check.entrypoint}",
            source_path=check.entrypoint,
        )
        return CheckRunOutcome(
            check_id=check.check_id,
            executed=False,
            skipped=True,
            skip_reason="internal_pending",
            findings=(finding,),
        )

    if check.entrypoint.startswith("TO_IMPLEMENT") or "(allowlisted" in check.entrypoint:
        finding = AuditFinding(
            finding_id=f"{check.check_id}:NOT_EXECUTABLE",
            check_id=check.check_id,
            severity=AuditSeverity.TRACKED_DEBT if mode is AuditMode.FAST else AuditSeverity.ERROR,
            category=check.category,
            message=f"Entrypoint not executable at HEAD: {check.entrypoint}",
            source_path=check.entrypoint,
        )
        return CheckRunOutcome(
            check_id=check.check_id,
            executed=False,
            skipped=True,
            skip_reason="non_executable_entrypoint",
            findings=(finding,),
        )

    try:
        result: SandboxResult = sandbox.run_check(check)
    except Exception as exc:  # noqa: BLE001 — audit boundary
        return CheckRunOutcome(
            check_id=check.check_id,
            executed=True,
            skipped=False,
            exit_code=3,
            findings=(
                AuditFinding(
                    finding_id=f"{check.check_id}:SANDBOX",
                    check_id=check.check_id,
                    severity=AuditSeverity.BLOCKER,
                    category=AuditCategory.INTERNAL,
                    message=f"Sandbox failure: {exc}",
                    source_path=check.entrypoint,
                ),
            ),
        )

    if result.exit_code == 0:
        return CheckRunOutcome(
            check_id=check.check_id,
            executed=True,
            skipped=False,
            exit_code=0,
            duration_ms=result.duration_ms,
            stdout_tail=result.stdout,
            stderr_tail=result.stderr,
        )

    return CheckRunOutcome(
        check_id=check.check_id,
        executed=True,
        skipped=False,
        exit_code=result.exit_code,
        duration_ms=result.duration_ms,
        stdout_tail=result.stdout,
        stderr_tail=result.stderr,
        findings=(_finding_from_failure(check, "Gate check failed (non-zero exit)"),),
    )
