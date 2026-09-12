# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Platform rollback provider — spine-directed intents (SELF-HEALING R2)."""

from __future__ import annotations

from intergrax.contracts.self_healing.workflow.context import SelfHealingWorkflowContext
from intergrax.contracts.self_healing.workflow.rollback import (
    SelfHealingRollbackDirective,
    SelfHealingRollbackPlan,
)


class PlatformDefaultRollbackProvider:
    provider_id = "platform.default.rollback"

    def plan_rollback(
        self,
        workflow_context: SelfHealingWorkflowContext,
    ) -> SelfHealingRollbackPlan:
        directives: list[SelfHealingRollbackDirective] = []
        for step_id in reversed(workflow_context.successful_step_ids):
            step = next(s for s in workflow_context.plan.steps if s.step_id == step_id)
            if step.rollback_reference is None:
                continue
            action = workflow_context.decision.proposed_actions[0]
            directives.append(
                SelfHealingRollbackDirective(
                    directive_id=f"rb_{step_id}",
                    operation_intent="self_healing.rollback.restore",
                    target_resource=action.target_resource,
                    rationale=f"rollback after failed validation for {step_id}",
                ),
            )
        if not directives:
            directives.append(
                SelfHealingRollbackDirective(
                    directive_id=f"rb_{workflow_context.workflow_id}",
                    operation_intent="self_healing.rollback.restore",
                    target_resource=workflow_context.decision.proposed_actions[0].target_resource,
                    rationale="default rollback directive",
                ),
            )
        return SelfHealingRollbackPlan(directives=tuple(directives))


__all__ = ["PlatformDefaultRollbackProvider"]
