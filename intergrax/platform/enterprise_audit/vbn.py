# © Artur Czarnecki. All rights reserved.

"""VBN-* check metadata and Annex D domain rollup."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditCategory, AuditFinding

_VBN_IDS = frozenset(
    {
        "VBN-INVENTORY-DRIFT",
        "VBN-CORE-HARDCODING",
        "VBN-SHADOW-SELECTION",
        "VBN-SILENT-FALLBACK",
        "VBN-CONTRACT-LEAKAGE",
        "VBN-SQLITE-EFFECTIVE",
        "VBN-OBS-OPS-COUPLING",
        "VBN-REPLACEABILITY-MATRIX",
        "VBN-REACHABILITY",
    }
)

_DOMAIN_ID_RE = re.compile(r"^D(0[1-9]|1[0-9]|2[0-9]|30|31)$")
_ALLOWED_STATUS = frozenset({"QUALIFIED", "NA_WITH_EVIDENCE", "GAP", "PARTIAL"})


def enrich_vbn_findings(findings: tuple[AuditFinding, ...]) -> tuple[AuditFinding, ...]:
    enriched: list[AuditFinding] = []
    for finding in findings:
        if finding.check_id in _VBN_IDS or finding.check_id.startswith("ENT-A-VBN-"):
            enriched.append(
                AuditFinding(
                    finding_id=finding.finding_id,
                    check_id=finding.check_id,
                    severity=finding.severity,
                    category=AuditCategory.VBN,
                    message=finding.message,
                    source_path=finding.source_path,
                    frz_mapping=finding.frz_mapping,
                    drift=finding.drift,
                )
            )
        else:
            enriched.append(finding)
    return tuple(enriched)


def validate_domain_report(report: dict, *, expected_sha: str | None = None) -> tuple[str, ...]:
    """Validate vbn_domains.v1.json shape and freshness (C008)."""
    errors: list[str] = []
    if report.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if int(report.get("domains_total", 0)) != 31:
        errors.append("domains_total must be 31")
    domains = report.get("domains")
    if not isinstance(domains, list) or len(domains) != 31:
        errors.append("domains must contain exactly 31 rows")
        return tuple(errors)
    seen: set[str] = set()
    for row in domains:
        if not isinstance(row, dict):
            errors.append("domain row must be object")
            continue
        domain_id = str(row.get("domain_id", ""))
        if not _DOMAIN_ID_RE.fullmatch(domain_id):
            errors.append(f"invalid domain_id: {domain_id}")
        if domain_id in seen:
            errors.append(f"duplicate domain_id: {domain_id}")
        seen.add(domain_id)
        status = str(row.get("status", ""))
        if status not in _ALLOWED_STATUS:
            errors.append(f"invalid status for {domain_id}: {status}")
    if len(seen) != 31:
        errors.append("expected D01..D31 unique domain_id values")
    qual_sha = str(report.get("qualification_sha", ""))
    if expected_sha and qual_sha and qual_sha != expected_sha:
        errors.append("qualification_sha stale vs expected HEAD")
    return tuple(errors)


def load_vbn_domain_report(
    repo_root: Path,
    output_dir: Path,
    *,
    expected_sha: str | None = None,
    allow_cached: bool = True,
) -> dict | None:
    cached = output_dir / "vbn_domains.v1.json"
    if allow_cached and cached.is_file():
        report = json.loads(cached.read_text(encoding="utf-8"))
        errors = validate_domain_report(report, expected_sha=expected_sha)
        if errors:
            return None
        return report
    script = repo_root / "scripts" / "enterprise_audit" / "vbn_domain_report.py"
    if not script.is_file():
        return None
    completed = subprocess.run(
        ["uv", "run", "python", str(script), "--output", str(cached)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not cached.is_file():
        return None
    report = json.loads(cached.read_text(encoding="utf-8"))
    errors = validate_domain_report(report, expected_sha=expected_sha)
    if errors:
        return None
    return report


def vbn_domain_rollup(
    findings: tuple[AuditFinding, ...],
    *,
    domain_report: dict | None = None,
) -> dict[str, int]:
    if domain_report and "domains" in domain_report:
        qualified = sum(1 for row in domain_report["domains"] if row.get("status") in {"QUALIFIED", "NA_WITH_EVIDENCE"})
        return {
            "domains_total": domain_report.get("domains_total", 31),
            "domains_accounted": qualified,
            "vbn_findings": len([f for f in findings if f.category is AuditCategory.VBN]),
        }
    vbn_findings = [f for f in findings if f.category is AuditCategory.VBN]
    return {
        "domains_total": 31,
        "domains_accounted": 0 if vbn_findings else 0,
        "vbn_findings": len(vbn_findings),
    }
