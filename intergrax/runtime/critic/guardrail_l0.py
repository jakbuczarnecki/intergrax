# © Artur Czarnecki. All rights reserved.

"""Compose vendor guardrail scan artifacts into CVL L0 verdicts (GR-INT.4)."""

from __future__ import annotations

from typing import Any

from intergrax.runtime.critic.contracts import CriticLayer, LayerVerdict


def guardrail_scan_from_context(context: dict[str, Any]) -> dict[str, Any] | None:
    raw = context.get("guardrail_scan")
    return raw if isinstance(raw, dict) else None


def merge_guardrail_l0(verdict: LayerVerdict, *, context: dict[str, Any]) -> LayerVerdict:
    """Augment L0 with guardrail scan metadata when Tier-3 passes ``guardrail_scan`` in critic context."""
    scan = guardrail_scan_from_context(context)
    if scan is None:
        return verdict
    if scan.get("allowed") is False:
        detail = str(scan.get("detail") or "guardrail scan blocked output")
        return LayerVerdict(
            layer=CriticLayer.L0_DETERMINISTIC,
            passed=False,
            score=0.0,
            errors=[*verdict.errors, f"guardrail_l0: {detail}"],
            warnings=list(verdict.warnings),
        )
    categories = scan.get("categories")
    warnings = list(verdict.warnings)
    if isinstance(categories, (list, tuple)) and categories:
        warnings.append(f"guardrail_l0: categories={list(categories)}")
    elif isinstance(categories, str) and categories:
        warnings.append(f"guardrail_l0: categories=[{categories}]")
    return verdict.model_copy(update={"warnings": warnings})
