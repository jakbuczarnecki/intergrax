# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.security.contracts import (
    SecurityScanInput,
    SecurityScanOutput,
    SecuritySummarizeFindingsInput,
    SecuritySummarizeFindingsOutput,
)
from intergrax.tools.providers.security.service import security_scan, security_summarize_findings


class SecurityScanHandler(ServiceToolHandler[SecurityScanInput, SecurityScanOutput]):
    _service = security_scan


class SecuritySummarizeFindingsHandler(
    ServiceToolHandler[SecuritySummarizeFindingsInput, SecuritySummarizeFindingsOutput]
):
    _service = security_summarize_findings
