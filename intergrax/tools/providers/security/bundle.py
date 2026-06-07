# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.security.contracts import (
    SecurityScanInput,
    SecurityScanOutput,
    SecuritySummarizeFindingsInput,
    SecuritySummarizeFindingsOutput,
)
from intergrax.tools.providers.security.handlers import SecurityScanHandler, SecuritySummarizeFindingsHandler
from intergrax.tools.providers.security.service import SECURITY_SCAN_TOOL_ID, SECURITY_SUMMARIZE_FINDINGS_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

SECURITY_BUNDLE_ID = "security"
SECURITY_TOOL_IDS: tuple[str, ...] = (SECURITY_SCAN_TOOL_ID, SECURITY_SUMMARIZE_FINDINGS_TOOL_ID)


def register_security_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=SECURITY_SCAN_TOOL_ID,
            name=SECURITY_SCAN_TOOL_ID,
            description="Run image or repository security scan via configured scanner backend.",
            description_short="Security scan (image/repo).",
            input_schema=SecurityScanInput,
            output_schema=SecurityScanOutput,
            error_mapping={},
            side_effects=False,
            category="security",
            risk_level=ToolRiskLevel.LOW,
            tags=("security", "scanner", "trivy", "semgrep"),
        ),
        SecurityScanHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=SECURITY_SUMMARIZE_FINDINGS_TOOL_ID,
            name=SECURITY_SUMMARIZE_FINDINGS_TOOL_ID,
            description="Aggregate security findings by severity (V-SEC promote gate helper).",
            description_short="Summarize scan findings.",
            input_schema=SecuritySummarizeFindingsInput,
            output_schema=SecuritySummarizeFindingsOutput,
            error_mapping={},
            side_effects=False,
            category="security",
            risk_level=ToolRiskLevel.LOW,
            tags=("security", "scanner", "summary"),
        ),
        SecuritySummarizeFindingsHandler(ctx),
    )
