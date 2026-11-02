# © Artur Czarnecki. All rights reserved.

"""Baseline load/store for enterprise audit drift."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import (
    AuditCategory,
    AuditFinding,
    AuditSeverity,
    DriftStatus,
)


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
            drift = DriftStatus.UNCHANGED
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
