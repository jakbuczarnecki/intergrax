# © Artur Czarnecki. All rights reserved.

"""Deterministic guardrail patterns — harness fallback when vendor SDK absent."""

from __future__ import annotations

import re

from intergrax.integrations.contracts.llm_guardrail import (
    GuardrailRiskLevel,
    GuardrailScanResult,
)

_INJECTION_PATTERNS: tuple[tuple[str, str, GuardrailRiskLevel], ...] = (
    ("ignore previous instructions", "prompt_injection", GuardrailRiskLevel.HIGH),
    ("ignore all prior", "prompt_injection", GuardrailRiskLevel.HIGH),
    ("system override", "prompt_injection", GuardrailRiskLevel.HIGH),
    ("jailbreak", "prompt_injection", GuardrailRiskLevel.HIGH),
    ("BLOCK_INPUT", "test_block", GuardrailRiskLevel.CRITICAL),
    ("BLOCK_OUTPUT", "test_block", GuardrailRiskLevel.CRITICAL),
)

_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "pii_ssn"),
    (re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"), "pii_credit_card"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "pii_email"),
)


def scan_patterns(
    text: str,
    *,
    mode: str,
    slug: str,
) -> GuardrailScanResult:
    lowered = text.lower()
    matched_rules: list[str] = []
    categories: list[str] = []
    highest = GuardrailRiskLevel.LOW

    for pattern, category, risk in _INJECTION_PATTERNS:
        if pattern.lower() in lowered:
            matched_rules.append(f"{category}:{pattern}")
            if category not in categories:
                categories.append(category)
            if _risk_rank(risk) > _risk_rank(highest):
                highest = risk

    for regex, category in _PII_PATTERNS:
        if regex.search(text):
            matched_rules.append(f"{category}:regex")
            if category not in categories:
                categories.append(category)
            if _risk_rank(GuardrailRiskLevel.MEDIUM) > _risk_rank(highest):
                highest = GuardrailRiskLevel.MEDIUM

    blocked = any(category in {"prompt_injection", "test_block"} for category in categories)

    return GuardrailScanResult(
        allowed=not blocked,
        risk_level=highest if categories else GuardrailRiskLevel.LOW,
        categories=tuple(categories),
        matched_rules=tuple(matched_rules),
        sanitized_text=text if not blocked else None,
        detail=f"{slug} pattern scan ({mode})",
        audit_payload={"engine": "pattern", "slug": slug, "mode": mode},
    )


def _risk_rank(level: GuardrailRiskLevel) -> int:
    order = {
        GuardrailRiskLevel.LOW: 0,
        GuardrailRiskLevel.MEDIUM: 1,
        GuardrailRiskLevel.HIGH: 2,
        GuardrailRiskLevel.CRITICAL: 3,
    }
    return order[level]
