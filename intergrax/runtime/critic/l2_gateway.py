# © Artur Czarnecki. All rights reserved.

"""L2 human verification gateway — routes to HITL (Phase CRIT-V-FOLLOWUP)."""

from __future__ import annotations

from intergrax.runtime.critic.contracts import CriticLayer, CriticRequest, LayerVerdict


class L2Gateway:
    """
    Authoritative human verification layer.

    Does not block synchronously on human input — returns a pending verdict that
    orchestration maps to ``ESCALATE_HITL``.
    """

    def verify(self, request: CriticRequest) -> LayerVerdict:
        scope = request.scope.value
        return LayerVerdict(
            layer=CriticLayer.L2_HUMAN,
            passed=False,
            warnings=[f"human_verification_required:{scope}"],
        )
