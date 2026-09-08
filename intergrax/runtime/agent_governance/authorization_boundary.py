# © Artur Czarnecki. All rights reserved.

"""Pre-execution authorization boundary for agent runtime governance."""

from __future__ import annotations

from intergrax.contracts.agent_runtime_governance import (
    ToolAuthorizationDecision,
    ToolAuthorizationDecisionState,
    ToolAuthorizationRequest,
)
from intergrax.runtime.agent_governance.errors import (
    ToolGovernanceApprovalRequiredError,
    ToolGovernanceDeniedError,
)
from intergrax.runtime.agent_governance.pipeline import AgentRuntimeGovernancePipeline
from intergrax.runtime.agent_governance.ports import AgentRuntimeGovernancePort


class AgentRuntimeGovernanceBoundary:
    """
    Shared authorization boundary evaluated before tool execution.

    Evaluation only — domain executors perform tool invocation after ALLOW.
    """

    def __init__(self, pipeline: AgentRuntimeGovernancePipeline) -> None:
        self._pipeline = pipeline

    def authorize_tool(
        self,
        request: ToolAuthorizationRequest,
    ) -> ToolAuthorizationDecision:
        decision = self._pipeline.evaluate(request)
        if decision.is_terminal_deny:
            raise ToolGovernanceDeniedError(
                run_id=str(request.run_id),
                agent_id=request.agent.agent_id,
                tool_id=request.tool_id,
                capability=request.capability,
                reason=decision.reason,
                policy_results=decision.policy_results,
            )
        if decision.requires_approval:
            approval_id = _extract_approval_id(decision.reason)
            raise ToolGovernanceApprovalRequiredError(
                run_id=str(request.run_id),
                agent_id=request.agent.agent_id,
                tool_id=request.tool_id,
                capability=request.capability,
                approval_id=approval_id,
                reason=decision.reason,
                policy_results=decision.policy_results,
            )
        return decision


def _extract_approval_id(reason: str) -> str:
    marker = "approval_id="
    if marker in reason:
        return reason.split(marker, 1)[1].split(";", 1)[0]
    return "unknown"
