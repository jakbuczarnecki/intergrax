# © Artur Czarnecki. All rights reserved.

"""Map harness kernel step records to governance outcomes without re-evaluating policy."""

from __future__ import annotations

from intergrax.contracts.agent_decision import AgentDecision
from intergrax.contracts.agent_run_enums import AgentRunErrorCode
from intergrax.contracts.step_execution import StepExecutionRecord
from intergrax.runtime.interrupts.handler import GovernanceResolution


def governance_resolution_from_kernel_step_record(
    record: StepExecutionRecord,
    agent_decision: AgentDecision,
) -> GovernanceResolution | None:
    """Preserve canonical ``policy_pre`` when HarnessKernel blocked step execution."""
    if record.outcome_applied:
        return None
    if record.error_code is not AgentRunErrorCode.POLICY_DENIED:
        return None
    policy_pre = record.policy_pre
    if policy_pre is None:
        return None
    return GovernanceResolution(
        policy_decision=policy_pre,
        agent_decision=agent_decision,
    )
