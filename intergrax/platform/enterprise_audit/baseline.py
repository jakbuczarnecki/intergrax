# © Artur Czarnecki. All rights reserved.

"""Baseline load/store for enterprise audit drift."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import (
    AuditCategory,
    AuditFinding,
    AuditSeverity,
    DriftStatus,
)

_SEVERITY_RANK = {
    AuditSeverity.INFO: 0,
    AuditSeverity.TRACKED_DEBT: 1,
    AuditSeverity.WARNING: 2,
    AuditSeverity.ENVIRONMENT: 3,
    AuditSeverity.ERROR: 4,
    AuditSeverity.BLOCKER: 5,
}


@dataclass(frozen=True, slots=True)
class EnterpriseAuditBaseline:
    accepted_baseline_sha: str
    findings: dict[str, dict]


def baseline_path(default_dir: Path) -> Path:
    return default_dir / "ENTERPRISE_AUDIT_BASELINE.json"


def load_baseline(path: Path) -> EnterpriseAuditBaseline | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EnterpriseAuditBaseline(
        accepted_baseline_sha=str(raw["accepted_baseline_sha"]),
        findings={k: v for k, v in raw.get("findings", {}).items()},
    )


def save_baseline(path: Path, baseline: EnterpriseAuditBaseline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "accepted_baseline_sha": baseline.accepted_baseline_sha,
        "findings": baseline.findings,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _message_fingerprint(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]


def _regressed(baseline_row: dict, finding: AuditFinding) -> bool:
    old_sev = AuditSeverity(str(baseline_row.get("severity", AuditSeverity.INFO)))
    if _SEVERITY_RANK.get(finding.severity, 0) > _SEVERITY_RANK.get(old_sev, 0):
        return True
    old_fp = str(baseline_row.get("message_fingerprint", ""))
    if old_fp and old_fp != _message_fingerprint(finding.message):
        return True
    old_schema = int(baseline_row.get("schema_version", 1))
    if finding.check_id and old_schema > 0:
        pass
    return False


def apply_drift(
    findings: tuple[AuditFinding, ...],
    baseline: EnterpriseAuditBaseline | None,
) -> tuple[AuditFinding, ...]:
    if baseline is None:
        return findings
    updated: list[AuditFinding] = []
    baseline_ids = set(baseline.findings)
    current_ids = {f.finding_id for f in findings}
    for finding in findings:
        if finding.finding_id in baseline.findings:
            row = baseline.findings[finding.finding_id]
            drift = DriftStatus.REGRESSED if _regressed(row, finding) else DriftStatus.UNCHANGED
        else:
            drift = DriftStatus.NEW
        updated.append(
            AuditFinding(
                finding_id=finding.finding_id,
                check_id=finding.check_id,
                severity=finding.severity,
                category=finding.category,
                message=finding.message,
                source_path=finding.source_path,
                frz_mapping=finding.frz_mapping,
                drift=drift,
            )
        )
    for resolved_id in baseline_ids - current_ids:
        old = baseline.findings[resolved_id]
        updated.append(
            AuditFinding(
                finding_id=resolved_id,
                check_id=str(old.get("check_id", "")),
                severity=AuditSeverity(old.get("severity", AuditSeverity.INFO)),
                category=AuditCategory(old.get("category", AuditCategory.ARCHITECTURE)),
                message="Resolved since baseline",
                drift=DriftStatus.RESOLVED,
            )
        )
    return tuple(updated)


def findings_to_baseline_rows(findings: tuple[AuditFinding, ...]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for finding in findings:
        rows[finding.finding_id] = {
            "check_id": finding.check_id,
            "severity": finding.severity.value,
            "message_fingerprint": _message_fingerprint(finding.message),
            "schema_version": 1,
        }
    return rows
