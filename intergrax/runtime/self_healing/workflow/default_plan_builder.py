# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Platform default healing plan builder (SELF-HEALING R2)."""

from __future__ import annotations

from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.decision import SelfHealingDecision
from intergrax.contracts.self_healing.workflow.plan import (
    SelfHealingPlan,
    SelfHealingRiskLevel,
    mint_self_healing_plan_id,
)
from intergrax.contracts.self_healing.workflow.step import SelfHealingStep


class PlatformDefaultSelfHealingPlanBuilder:
    builder_id = "platform.default.plan_builder"

    def build_plan(
        self,
        decision: SelfHealingDecision,
        context: SelfHealingContext,
    ) -> SelfHealingPlan:
        steps: list[SelfHealingStep] = []
        seq = 0
        for action in decision.proposed_actions:
            steps.append(
                SelfHealingStep(
                    step_id=f"{decision.decision_id}_step_{seq}",
                    operation_intent=action.action_type,
                    required_capability=action.operation_kind,
                    sequence_number=seq,
                    validation_requirements=("metrics.stable",),
                    rollback_reference=f"rollback_{action.action_type}",
                ),
            )
            seq += 1
        steps.append(
            SelfHealingStep(
                step_id=f"{decision.decision_id}_wait",
                operation_intent="self_healing.workflow.wait_stabilization",
                required_capability="workflow.wait",
                sequence_number=seq,
                validation_requirements=(),
                rollback_reference=None,
            ),
        )
        seq += 1
        steps.append(
            SelfHealingStep(
                step_id=f"{decision.decision_id}_validate",
                operation_intent="self_healing.workflow.validate",
                required_capability="workflow.validate",
                sequence_number=seq,
                validation_requirements=("evidence.required",),
                rollback_reference=None,
            ),
        )
        risk = SelfHealingRiskLevel.HIGH if context.constraints.production_target else SelfHealingRiskLevel.MEDIUM
        return SelfHealingPlan(
            plan_id=mint_self_healing_plan_id(),
            plan_version="1",
            strategy_id=decision.strategy_id,
            tenant_id=context.tenant_id,
            scope=context.diagnostic_investigation.problem_id,
            steps=tuple(steps),
            validation_policy_id="platform.default.validation",
            rollback_policy_id="platform.default.rollback",
            evidence_refs=decision.evidence_refs,
            risk_level=risk,
        )


__all__ = ["PlatformDefaultSelfHealingPlanBuilder"]
