# © Artur Czarnecki. All rights reserved.

"""TEN-* check metadata enrichment (13 controls)."""

from __future__ import annotations

from intergrax.platform.enterprise_audit.contracts import AuditCategory, AuditFinding

_TEN_IDS = frozenset(
    {
        "TEN-IDENTITY",
        "TEN-PROPAGATION",
        "TEN-AUTHORITY",
        "TEN-STATE",
        "TEN-PROVIDER",
        "TEN-CONFIG",
        "TEN-CREDENTIALS",
        "TEN-EVIDENCE",
        "TEN-TRACE",
        "TEN-RECOVERY",
        "TEN-ASYNC",
        "TEN-CROSS-TENANT",
        "TEN-FAIL-CLOSED",
    }
)


def enrich_tenant_findings(findings: list[AuditFinding]) -> tuple[AuditFinding, ...]:
    enriched: list[AuditFinding] = []
    for finding in findings:
        if finding.check_id in _TEN_IDS or finding.check_id.startswith("ENT-A-TEN-"):
            enriched.append(
                AuditFinding(
                    finding_id=finding.finding_id,
                    check_id=finding.check_id,
                    severity=finding.severity,
                    category=AuditCategory.TENANT,
                    message=finding.message,
                    source_path=finding.source_path,
                    frz_mapping=finding.frz_mapping,
                    drift=finding.drift,
                )
            )
        else:
            enriched.append(finding)
    return tuple(enriched)
