# © Artur Czarnecki. All rights reserved.

"""Utility function for harness outcome signals (Phase W-ADAPT-1.8)."""

from __future__ import annotations

from intergrax.runtime.adaptive.contracts import UtilityWeights


def normalize_latency(latency_ms: int, *, latency_slo_ms: int) -> float:
    """Map latency to [0, 1] against an SLO ceiling."""
    if latency_slo_ms <= 0:
        return 0.0
    return min(1.0, max(0.0, latency_ms / latency_slo_ms))


def regression_penalty(regression_flags: list[str]) -> float:
    """Aggregate regression flags into a bounded penalty in [0, 1]."""
    if not regression_flags:
        return 0.0
    return min(1.0, 0.25 * len(regression_flags))


def compute_utility(
    *,
    quality_score: float,
    cost_normalized: float,
    latency_ms: int,
    hitl_interventions: int,
    regression_flags: list[str],
    business_outcome: float | None = None,
    weights: UtilityWeights | None = None,
) -> float:
    """Compute composite utility U per AHIA §10.2."""
    resolved = weights or UtilityWeights()
    latency_penalty = normalize_latency(latency_ms, latency_slo_ms=resolved.latency_slo_ms)
    cost_penalty = max(0.0, cost_normalized - 1.0)
    hitl_penalty = min(1.0, hitl_interventions / max(1, resolved.max_hitl_interventions))
    reg_penalty = regression_penalty(regression_flags)
    business_bonus = business_outcome or 0.0

    utility = (
        resolved.w_quality * quality_score
        - resolved.w_cost * cost_penalty
        - resolved.w_latency * latency_penalty
        - resolved.w_hitl * hitl_penalty
        - resolved.w_regression * reg_penalty
        + resolved.w_business * business_bonus
    )
    return max(-1.0, min(1.0, utility))
