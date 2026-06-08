# © Artur Czarnecki. All rights reserved.

"""Maps critic verdicts to orchestration and policy actions (Phase CRIT-V-FOLLOWUP)."""

from __future__ import annotations

from typing import Any, Mapping

from intergrax.runtime.critic.contracts import CriticAction, CriticLayer, CriticVerdict


def borderline_l1_score(
    verdict: CriticVerdict,
    *,
    threshold: float,
    margin: float,
) -> bool:
    """True when L1 semantic score is within ``margin`` below the pass threshold."""
    for layer in verdict.layers:
        if layer.layer is not CriticLayer.L1_SEMANTIC or layer.score is None:
            continue
        return threshold - margin <= layer.score < threshold
    return False


def resolve_critic_action(
    verdict: CriticVerdict,
    *,
    governance: Mapping[str, Any] | None = None,
    judge_threshold: float = 0.75,
    l2_borderline_margin: float = 0.05,
    l2_human_required: bool = False,
) -> CriticAction:
    """
    Combine orchestrator recommendation with Tier-3 ``critic_governance`` fragment.

    Borderline L1 scores escalate to HITL when ``l2_human_required`` is enabled.
    """
    if verdict.passed:
        return CriticAction.CONTINUE

    fragment = dict(governance or {})
    threshold = float(fragment.get("judge_threshold", judge_threshold))
    margin = float(fragment.get("l2_borderline_margin", l2_borderline_margin))
    require_l2 = bool(fragment.get("l2_human_required", l2_human_required))

    if require_l2 and borderline_l1_score(verdict, threshold=threshold, margin=margin):
        return CriticAction.ESCALATE_HITL

    if fragment.get("require_critic_on_completion") and not verdict.passed:
        if verdict.recommended_action is CriticAction.CONTINUE:
            return CriticAction.FAIL
        return verdict.recommended_action

    return verdict.recommended_action


def critic_governance_from_fragment(fragment: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize policy bundle ``critic_governance`` for runtime context."""
    if fragment is None:
        return {}
    return dict(fragment)
