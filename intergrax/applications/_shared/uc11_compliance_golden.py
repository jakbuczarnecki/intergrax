# © Artur Czarnecki. All rights reserved.

"""UC-11 golden compliance helpers for Tier-3 product hosts (ACP-CLOSE-ORG-2)."""

from __future__ import annotations

from intergrax.agents.compliance_summary import build_compliance_summary
from intergrax.agents.run_environment import EffectiveAgentRunEnvironment
from intergrax.contracts.agent_run import ComplianceSummary
from intergrax.contracts.agent_run_trace import AgentRunTrace
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.runtime.kernel.step_kernel import HarnessKernel, StepKernelContext
from intergrax.runtime.policy.policy_engine import PolicyEngine


async def run_uc11_kernel_happy_path_step(
    merged: EffectiveAgentRunEnvironment,
    *,
    channel: str = "chat",
) -> AgentRunTrace:
    """
    Execute one harness step on the happy UC-11 path.

    Asserts no org policy denial on allowed channel and returns the run trace.
    """
    if merged.organizational is None:
        raise AssertionError("UC-11 golden path requires merged.organizational")

    step_ctx = AgentStepContext(
        step_index=0,
        run_id="uc11-golden",
        agent_id=merged.agent_id,
        contract_id=merged.contract_id,
        metadata={"channel": channel},
    )
    kernel_ctx = StepKernelContext(
        agent_id=merged.agent_id,
        run_id="uc11-golden",
        tenant_id=merged.tenant_id,
        policy_engine=PolicyEngine(),
        organizational=merged.organizational,
    )
    outcome = StepOutcome.continue_with({"phase": "uc11_golden"})
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    if record.error_code is not None:
        raise AssertionError(f"UC-11 golden kernel step failed: {record.error_code}")
    if record.step_record is not None:
        for verdict in record.step_record.policy_verdicts:
            if verdict.policy_rule_id.startswith("org.") and verdict.action == PolicyAction.DENY:
                raise AssertionError(
                    f"UC-11 golden path org denial: {verdict.policy_rule_id} — {verdict.reason}",
                )
    return kernel_ctx.run_trace


def assert_golden_compliance_zero_denials(trace: AgentRunTrace) -> ComplianceSummary:
    """Golden eval — zero POLICY_DENIED on happy path (architecture §39.5)."""
    summary = build_compliance_summary(trace)
    if summary.deny_count != 0:
        raise AssertionError(
            f"UC-11 golden compliance expected zero denials, got {summary.deny_count}: "
            f"{summary.rules_triggered}",
        )
    return summary
