# © Artur Czarnecki. All rights reserved.

"""Maps critic verdicts to orchestration and policy actions (Phase CRIT-V-FOLLOWUP)."""

from __future__ import annotations

from typing import Mapping

from intergrax.runtime.critic.contracts import CriticAction, CriticVerdict


def resolve_critic_action(
    verdict: CriticVerdict,
    *,
    governance: Mapping[str, object] | None = None,
) -> CriticAction:
    """Combine orchestrator recommendation with Tier-3 ``critic_governance`` fragment."""
    if verdict.passed:
        return CriticAction.CONTINUE

    fragment = dict(governance or {})
    if fragment.get("require_critic_on_completion") and not verdict.passed:
        if verdict.recommended_action is CriticAction.CONTINUE:
            return CriticAction.FAIL
        return verdict.recommended_action

    return verdict.recommended_action


def critic_governance_from_fragment(fragment: Mapping[str, object] | None) -> dict[str, object]:
    """Normalize policy bundle ``critic_governance`` for runtime context."""
    if fragment is None:
        return {}
    return dict(fragment)
