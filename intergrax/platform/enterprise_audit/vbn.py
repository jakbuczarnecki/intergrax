# © Artur Czarnecki. All rights reserved.

"""VBN-* check metadata and Annex D domain rollup."""

from __future__ import annotations

import json
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


def load_vbn_domain_report(repo_root: Path, output_dir: Path) -> dict | None:
    cached = output_dir / "vbn_domains.v1.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    script = repo_root / "scripts" / "enterprise_audit" / "vbn_domain_report.py"
    if not script.is_file():
        return None
    completed = subprocess.run(
        ["uv", "run", "python", str(script), "--output", str(output_dir / "vbn_domains.v1.json")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not cached.is_file():
        return None
    return json.loads(cached.read_text(encoding="utf-8"))


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
