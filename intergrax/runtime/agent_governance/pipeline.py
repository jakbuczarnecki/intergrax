# © Artur Czarnecki. All rights reserved.

"""Deterministic governance pipeline: capability → policy → risk → decision."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from intergrax.contracts.agent_runtime_governance import (
    CapabilityGrantResolverPort,
    ToolAuthorizationDecision,
    ToolAuthorizationDecisionState,
    ToolAuthorizationRequest,
)
from intergrax.runtime.agent_governance.approval_boundary import AgentRuntimeApprovalBoundary
from intergrax.runtime.agent_governance.audit import GovernanceAuditRecorder
from intergrax.runtime.agent_governance.capability_resolver import require_capability_granted
from intergrax.runtime.agent_governance.policy_engine import AgentRuntimePolicyEngine


class AgentRuntimeGovernancePipeline:
    """
    Deterministic governance pipeline:

    Agent Action Request → Capability Check → Policy Evaluation → Decision
    """

    def __init__(
        self,
        *,
        capability_resolver: CapabilityGrantResolverPort,
        policy_engine: AgentRuntimePolicyEngine,
        audit_recorder: GovernanceAuditRecorder,
        approval_boundary: AgentRuntimeApprovalBoundary | None = None,
        approval_ttl: timedelta = timedelta(hours=1),
    ) -> None:
        self._capability_resolver = capability_resolver
        self._policy_engine = policy_engine
        self._audit_recorder = audit_recorder
        self._approval_boundary = approval_boundary
        self._approval_ttl = approval_ttl

    def evaluate(self, request: ToolAuthorizationRequest) -> ToolAuthorizationDecision:
        grant = self._capability_resolver.resolve_grant(request.agent)
        require_capability_granted(
            grant=grant,
            capability=request.capability,
            run_id=str(request.run_id),
            agent_id=request.agent.agent_id,
            tool_id=request.tool_id,
        )

        decision = self._policy_engine.evaluate(request)

        if decision.requires_approval and self._approval_boundary is not None:
            expires_at = datetime.now(timezone.utc) + self._approval_ttl
            approval = self._approval_boundary.create_approval_request(
                request,
                expires_at=expires_at,
            )
            decision = ToolAuthorizationDecision(
                decision=ToolAuthorizationDecisionState.REQUIRE_APPROVAL,
                reason=f"{decision.reason};approval_id={approval.approval_id}",
                policy_results=decision.policy_results,
            )

        self._audit_recorder.record_decision(request, decision)
        return decision
