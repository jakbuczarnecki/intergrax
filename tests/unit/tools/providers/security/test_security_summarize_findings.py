# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.tools.providers.security.contracts import SecurityFindingOutput, SecuritySummarizeFindingsInput
from intergrax.tools.providers.security.service import security_summarize_findings
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


def test_security_summarize_findings_counts_severity() -> None:
    out = security_summarize_findings(
        ToolWiringContext(),
        SecuritySummarizeFindingsInput(
            findings=[
                SecurityFindingOutput(
                    id="f-1",
                    severity="critical",
                    title="RCE",
                    resource="img",
                    detail="",
                ),
                SecurityFindingOutput(
                    id="f-2",
                    severity="high",
                    title="CVE",
                    resource="repo",
                    detail="",
                ),
                SecurityFindingOutput(
                    id="f-3",
                    severity="high",
                    title="CVE-2",
                    resource="repo",
                    detail="",
                ),
            ]
        ),
    )
    assert out.total == 3
    assert out.critical_count == 1
    assert out.high_count == 2
