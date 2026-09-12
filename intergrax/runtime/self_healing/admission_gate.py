# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Policy self-healing admission — confidence is never authorization (SELF-HEALING R1)."""

from __future__ import annotations

from intergrax.contracts.self_healing.decision import SelfHealingDecision
from intergrax.contracts.self_healing.governance import (
    SelfHealingAdmissionContext,
    SelfHealingAdmissionDecision,
    SelfHealingAdmissionVerdict,
)


class PolicySelfHealingAdmissionGate:
    """Fail-closed admission for self-healing decisions."""

    _ESCALATION_ACTION = "self_healing.human.escalation"

    def evaluate(
        self,
        decision: SelfHealingDecision,
        context: SelfHealingAdmissionContext,
    ) -> SelfHealingAdmissionDecision:
        first_action = decision.proposed_actions[0].action_type if decision.proposed_actions else ""
        if not context.tenant_id.strip():
            return SelfHealingAdmissionDecision(
                verdict=SelfHealingAdmissionVerdict.DENY,
                reason="tenant scope missing",
                decision_id=context.decision_id,
            )
        if not context.governance_approved:
            return SelfHealingAdmissionDecision(
                verdict=SelfHealingAdmissionVerdict.DENY,
                reason="decision governance rejected self-healing proposal",
                decision_id=context.decision_id,
            )
        if decision.required_approval or context.production_target:
            if context.human_approval_granted and context.approval_id:
                return SelfHealingAdmissionDecision(
                    verdict=SelfHealingAdmissionVerdict.ALLOW,
                    reason="self-healing action with recorded human approval",
                    approval_id=context.approval_id,
                    decision_id=context.decision_id,
                )
            return SelfHealingAdmissionDecision(
                verdict=SelfHealingAdmissionVerdict.REQUIRES_APPROVAL,
                reason="self-healing action requires human approval",
                decision_id=context.decision_id,
            )
        if first_action == self._ESCALATION_ACTION:
            return SelfHealingAdmissionDecision(
                verdict=SelfHealingAdmissionVerdict.REQUIRES_APPROVAL,
                reason="human escalation always requires approval",
                decision_id=context.decision_id,
            )
        return SelfHealingAdmissionDecision(
            verdict=SelfHealingAdmissionVerdict.ALLOW,
            reason="policy allow",
            decision_id=context.decision_id,
        )


__all__ = ["PolicySelfHealingAdmissionGate"]
