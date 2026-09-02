# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Domain-neutral guardrail scan assessment for verification (DS-VER-STAGE-GR).

Quality/validity guardrail outcomes only — not authorization or policy decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.contracts.llm_guardrail import GuardrailScanResult


@dataclass(frozen=True, slots=True)
class GuardrailScanAssessment:
    """Deterministic assessment of one normalized guardrail scan result."""

    passed: bool
    detail: str
    categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("GuardrailScanAssessment.passed must be bool")
        if type(self.detail) is not str:
            raise TypeError("GuardrailScanAssessment.detail must be str")
        if type(self.categories) is not tuple:
            raise TypeError("GuardrailScanAssessment.categories must be tuple")


def assess_guardrail_scan(scan: GuardrailScanResult) -> GuardrailScanAssessment:
    """Assess one quality guardrail scan without policy or authorization semantics."""
    if type(scan) is not GuardrailScanResult:
        raise TypeError("scan must be GuardrailScanResult")
    if not scan.allowed:
        detail = scan.detail.strip() if scan.detail else "guardrail scan blocked output"
        return GuardrailScanAssessment(
            passed=False,
            detail=detail,
            categories=scan.categories,
        )
    return GuardrailScanAssessment(
        passed=True,
        detail="",
        categories=scan.categories,
    )
