from __future__ import annotations

from intergrax.platform.enterprise_audit.contracts import AuditExecutionManifest, AuditMode


def sample_full_execution_manifest(*, qualification_sha: str = "a" * 40) -> AuditExecutionManifest:
    return AuditExecutionManifest(
        audit_mode=AuditMode.FULL,
        qualification_sha=qualification_sha,
        audit_execution_sha=qualification_sha,
        checks_planned=85,
        checks_executed=85,
        checks_skipped=0,
        coverage_label="FULL",
        certifying_run=True,
    )
