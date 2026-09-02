# © Artur Czarnecki. All rights reserved.

"""Compose vendor guardrail scan artifacts into CVL L0 verdicts (GR-INT.4)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.guardrail_verification import assess_guardrail_scan
from intergrax.integrations.contracts.llm_guardrail import GuardrailScanResult
from intergrax.runtime.critic.contracts import CriticLayer, LayerVerdict


def guardrail_scan_from_context(context: dict[str, Any]) -> dict[str, Any] | None:
    raw = context.get("guardrail_scan")
    return raw if isinstance(raw, dict) else None


def _guardrail_scan_from_mapping(scan: dict[str, Any]) -> GuardrailScanResult:
    allowed = scan.get("allowed")
    resolved_allowed = True if allowed is None else bool(allowed)
    detail = str(scan.get("detail") or "")
    categories_raw = scan.get("categories")
    categories: tuple[str, ...] = ()
    if isinstance(categories_raw, (list, tuple)):
        categories = tuple(str(category) for category in categories_raw)
    elif isinstance(categories_raw, str) and categories_raw:
        categories = (categories_raw,)
    return GuardrailScanResult(
        allowed=resolved_allowed,
        detail=detail,
        categories=categories,
    )


def merge_guardrail_l0(verdict: LayerVerdict, *, context: dict[str, Any]) -> LayerVerdict:
    """Augment L0 with guardrail scan metadata when Tier-3 passes ``guardrail_scan`` in critic context."""
    scan_mapping = guardrail_scan_from_context(context)
    if scan_mapping is None:
        return verdict
    assessment = assess_guardrail_scan(_guardrail_scan_from_mapping(scan_mapping))
    if not assessment.passed:
        detail = assessment.detail or "guardrail scan blocked output"
        return LayerVerdict(
            layer=CriticLayer.L0_DETERMINISTIC,
            passed=False,
            score=0.0,
            errors=[*verdict.errors, f"guardrail_l0: {detail}"],
            warnings=list(verdict.warnings),
        )
    warnings = list(verdict.warnings)
    if assessment.categories:
        warnings.append(f"guardrail_l0: categories={list(assessment.categories)}")
    return verdict.model_copy(update={"warnings": warnings})
