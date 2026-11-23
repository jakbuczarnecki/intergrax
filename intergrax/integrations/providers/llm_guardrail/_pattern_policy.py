# © Artur Czarnecki. All rights reserved.

"""Pattern-engine fallback policy for guardrail vendors (FC-3)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.llm_guardrail import GuardrailScanResult
from intergrax.integrations.providers.llm_guardrail._pattern_scanner import scan_patterns


def pattern_fallback_scan(
    text: str,
    *,
    mode: str,
    slug: str,
    allow_pattern_fallback: bool,
) -> GuardrailScanResult:
    if not allow_pattern_fallback:
        raise IntegrationConfigurationError(
            f"{slug!r} vendor engine is unavailable and pattern fallback is not permitted"
        )
    return scan_patterns(text, mode=mode, slug=slug)
