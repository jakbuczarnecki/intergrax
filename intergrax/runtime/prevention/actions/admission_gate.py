# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Policy preventive action admission — confidence is never authorization (PREVENTIVE R7)."""

from __future__ import annotations

from intergrax.contracts.preventive.actions.action_type import PreventiveActionType
from intergrax.contracts.preventive.actions.admission import (
    PreventiveActionAdmissionContext,
    PreventiveActionAdmissionDecision,
    PreventiveActionAdmissionVerdict,
)
from intergrax.contracts.preventive.actions.proposal import PreventiveActionProposal


class PolicyPreventiveActionAdmissionGate:
    """Fail-closed admission for preventive proposals."""

    _HIGH_IMPACT_TYPES = frozenset(
        {
            PreventiveActionType.CONFIGURATION_UPDATE.qualified_id,
            f"{PreventiveActionType.CONFIGURATION_UPDATE.namespace}.{PreventiveActionType.CONFIGURATION_UPDATE.name}",
            PreventiveActionType.CAPACITY_SCALE.qualified_id,
            f"{PreventiveActionType.CAPACITY_SCALE.namespace}.{PreventiveActionType.CAPACITY_SCALE.name}",
            PreventiveActionType.INTEGRATION_PAUSE.qualified_id,
            f"{PreventiveActionType.INTEGRATION_PAUSE.namespace}.{PreventiveActionType.INTEGRATION_PAUSE.name}",
        },
    )

    def evaluate(
        self,
        proposal: PreventiveActionProposal,
        context: PreventiveActionAdmissionContext,
    ) -> PreventiveActionAdmissionDecision:
        if proposal.tenant_id != context.tenant_id:
            return PreventiveActionAdmissionDecision(
                verdict=PreventiveActionAdmissionVerdict.DENY,
                reason="tenant scope mismatch",
                decision_id=context.decision_id,
            )
        if not context.governance_approved:
            return PreventiveActionAdmissionDecision(
                verdict=PreventiveActionAdmissionVerdict.DENY,
                reason="decision governance rejected preventive proposal",
                decision_id=context.decision_id,
            )
        if proposal.action_type in self._HIGH_IMPACT_TYPES or context.production_target:
            if context.human_approval_granted and context.approval_id:
                return PreventiveActionAdmissionDecision(
                    verdict=PreventiveActionAdmissionVerdict.ALLOW,
                    reason="high-impact preventive action with recorded human approval",
                    approval_id=context.approval_id,
                    decision_id=context.decision_id,
                )
            return PreventiveActionAdmissionDecision(
                verdict=PreventiveActionAdmissionVerdict.REQUIRES_APPROVAL,
                reason="high-impact preventive action requires human approval",
                decision_id=context.decision_id,
            )
        if proposal.action_type in {
            PreventiveActionType.HUMAN_REVIEW.qualified_id,
            f"{PreventiveActionType.HUMAN_REVIEW.namespace}.{PreventiveActionType.HUMAN_REVIEW.name}",
        }:
            return PreventiveActionAdmissionDecision(
                verdict=PreventiveActionAdmissionVerdict.REQUIRES_APPROVAL,
                reason="human.review action always requires approval",
                decision_id=context.decision_id,
            )
        return PreventiveActionAdmissionDecision(
            verdict=PreventiveActionAdmissionVerdict.ALLOW,
            reason="policy allow",
            decision_id=context.decision_id,
        )


__all__ = ["PolicyPreventiveActionAdmissionGate"]
