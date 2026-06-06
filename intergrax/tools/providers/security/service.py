# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Security scanner catalog tool services."""

from __future__ import annotations

from intergrax.integrations.contracts.security_scanner import SecurityScannerBackend
from intergrax.tools.providers.security.contracts import SecurityFindingOutput, SecurityScanInput, SecurityScanOutput
from intergrax.tools.registry.wiring import ToolWiringContext

SECURITY_SCAN_TOOL_ID = "security.scan"


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
