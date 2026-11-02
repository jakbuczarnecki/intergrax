# © Artur Czarnecki. All rights reserved.

"""VBN-* check metadata and Annex D domain rollup."""

from __future__ import annotations

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


def vbn_domain_rollup(findings: tuple[AuditFinding, ...]) -> dict[str, int]:
    vbn_findings = [f for f in findings if f.category is AuditCategory.VBN]
    return {
        "domains_total": 31,
        "domains_accounted": 31 if not vbn_findings else max(0, 31 - len(vbn_findings)),
        "vbn_findings": len(vbn_findings),
    }
