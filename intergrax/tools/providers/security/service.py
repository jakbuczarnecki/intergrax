# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Security scanner catalog tool services."""

from __future__ import annotations

from intergrax.integrations.contracts.security_scanner import SecurityScannerBackend
from intergrax.tools.providers.security.contracts import (
    SecurityFindingOutput,
    SecurityScanInput,
    SecurityScanOutput,
    SecuritySeverityCountOutput,
    SecuritySummarizeFindingsInput,
    SecuritySummarizeFindingsOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

SECURITY_SCAN_TOOL_ID = "security.scan"
SECURITY_SUMMARIZE_FINDINGS_TOOL_ID = "security.summarize_findings"


def _require_scanner(ctx: ToolWiringContext) -> SecurityScannerBackend:
    backend = ctx.security_scanner
    if backend is None:
        raise RuntimeError("security_scanner_not_configured")
    return backend


def security_scan(ctx: ToolWiringContext, params: SecurityScanInput) -> SecurityScanOutput:
    backend = _require_scanner(ctx)
    if params.scan_type == "image":
        report = backend.scan_image(params.target)
    else:
        report = backend.scan_repo(params.target)
    findings = [
        SecurityFindingOutput(
            id=item.id,
            severity=item.severity,
            title=item.title,
            resource=item.resource,
            detail=item.detail,
        )
        for item in report.findings
    ]
    return SecurityScanOutput(target=report.target, status=report.status, findings=findings)


def security_summarize_findings(
    ctx: ToolWiringContext,
    params: SecuritySummarizeFindingsInput,
) -> SecuritySummarizeFindingsOutput:
    del ctx
    counts: dict[str, int] = {}
    for item in params.findings:
        key = item.severity.strip().lower() or "unknown"
        counts[key] = counts.get(key, 0) + 1
    by_severity = [
        SecuritySeverityCountOutput(severity=severity, count=count)
        for severity, count in sorted(counts.items())
    ]
    return SecuritySummarizeFindingsOutput(
        total=len(params.findings),
        by_severity=by_severity,
        critical_count=counts.get("critical", 0),
        high_count=counts.get("high", 0),
    )
