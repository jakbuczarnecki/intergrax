from __future__ import annotations

from intergrax.platform.enterprise_audit.contracts import AuditExecutionManifest, AuditMode, CheckRunOutcome
from intergrax.platform.enterprise_audit.digests import manifest_digest
from intergrax.platform.enterprise_audit.manifest import load_default_manifest


def sample_full_execution_manifest(
    *,
    qualification_sha: str = "a" * 40,
    include_closed_world_outcomes: bool = True,
) -> AuditExecutionManifest:
    outcomes: tuple[CheckRunOutcome, ...] = ()
    planned = 85
    executed = 85
    if include_closed_world_outcomes:
        checks = load_default_manifest()
        outcomes = tuple(
            CheckRunOutcome(
                check_id=check.check_id,
                executed=True,
                skipped=False,
                exit_code=0,
            )
            for check in checks
        )
    return AuditExecutionManifest(
        audit_mode=AuditMode.FULL,
        qualification_sha=qualification_sha,
        audit_execution_sha=qualification_sha,
        checks_planned=planned,
        checks_executed=executed,
        checks_skipped=0,
        coverage_label="FULL",
        certifying_run=True,
        run_status="completed",
        check_manifest_digest=manifest_digest(),
        outcomes=outcomes,
    )
