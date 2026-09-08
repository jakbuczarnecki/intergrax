# © Artur Czarnecki. All rights reserved.

"""Composable, deterministic policy evaluation for agent runtime governance."""

from __future__ import annotations

from intergrax.contracts.agent_runtime_governance import (
    AgentRuntimePolicyProvider,
    PolicyEvaluationResult,
    ToolAuthorizationDecision,
    ToolAuthorizationDecisionState,
    ToolAuthorizationRequest,
    ToolAuthorizationRiskLevel,
)

_DECISION_PRECEDENCE: tuple[ToolAuthorizationDecisionState, ...] = (
    ToolAuthorizationDecisionState.DENY,
    ToolAuthorizationDecisionState.REQUIRE_APPROVAL,
    ToolAuthorizationDecisionState.ALLOW,
)


def _merge_decisions(
    results: tuple[PolicyEvaluationResult, ...],
) -> ToolAuthorizationDecision:
    if not results:
        return ToolAuthorizationDecision(
            decision=ToolAuthorizationDecisionState.DENY,
            reason="no_policy_providers_configured",
            policy_results=(),
        )

    decision_rank = {state: index for index, state in enumerate(_DECISION_PRECEDENCE)}
    winning = min(results, key=lambda item: decision_rank[item.decision])
    contributing = tuple(
        item for item in results if item.decision is winning.decision
    )
    reasons = "; ".join(item.reason for item in contributing)
    return ToolAuthorizationDecision(
        decision=winning.decision,
        reason=reasons,
        policy_results=results,
    )


class AgentRuntimePolicyEngine:
    """
    Plugin-friendly policy engine with deterministic ordering.

    Policies are evaluated in registration order; the most restrictive
    decision wins (DENY > REQUIRE_APPROVAL > ALLOW).
    """

    def __init__(self, providers: tuple[AgentRuntimePolicyProvider, ...] = ()) -> None:
        self._providers = tuple(providers)

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(provider.policy_id for provider in self._providers)

    def evaluate(self, request: ToolAuthorizationRequest) -> ToolAuthorizationDecision:
        results: list[PolicyEvaluationResult] = []
        for provider in self._providers:
            results.append(provider.evaluate(request))
        return _merge_decisions(tuple(results))


class AllowAllPolicyProvider:
    """Baseline permissive policy — explicit opt-in, no implicit fallback."""

    @property
    def policy_id(self) -> str:
        return "governance.allow_all"

    def evaluate(self, request: ToolAuthorizationRequest) -> PolicyEvaluationResult:
        return PolicyEvaluationResult(
            policy_id=self.policy_id,
            decision=ToolAuthorizationDecisionState.ALLOW,
            reason="allow_all_policy",
        )


class DenyCapabilityPolicyProvider:
    """Denies specific capabilities regardless of grant."""

    def __init__(self, *, denied_capabilities: frozenset[str]) -> None:
        self._denied = denied_capabilities

    @property
    def policy_id(self) -> str:
        return "governance.deny_capability"

    def evaluate(self, request: ToolAuthorizationRequest) -> PolicyEvaluationResult:
        if request.capability in self._denied:
            return PolicyEvaluationResult(
                policy_id=self.policy_id,
                decision=ToolAuthorizationDecisionState.DENY,
                reason=f"capability_denied:{request.capability}",
            )
        return PolicyEvaluationResult(
            policy_id=self.policy_id,
            decision=ToolAuthorizationDecisionState.ALLOW,
            reason="capability_not_denied",
        )


class FinancialApprovalPolicyProvider:
    """
    Example enterprise policy: financial actions require human approval.

    Demonstrates plugin-friendly, independently deployable policy.
    """

    _FINANCIAL_CAPABILITIES: frozenset[str] = frozenset(
        {
            "approve_payment",
            "transfer_funds",
            "issue_refund",
        }
    )

    @property
    def policy_id(self) -> str:
        return "governance.financial_approval"

    def evaluate(self, request: ToolAuthorizationRequest) -> PolicyEvaluationResult:
        if request.capability in self._FINANCIAL_CAPABILITIES:
            if request.approval_evidence_ref:
                return PolicyEvaluationResult(
                    policy_id=self.policy_id,
                    decision=ToolAuthorizationDecisionState.ALLOW,
                    reason="financial_approval_evidence_present",
                )
            return PolicyEvaluationResult(
                policy_id=self.policy_id,
                decision=ToolAuthorizationDecisionState.REQUIRE_APPROVAL,
                reason=f"financial_capability_requires_approval:{request.capability}",
            )
        return PolicyEvaluationResult(
            policy_id=self.policy_id,
            decision=ToolAuthorizationDecisionState.ALLOW,
            reason="not_financial_capability",
        )


class HighRiskApprovalPolicyProvider:
    """Requires approval for HIGH and CRITICAL risk classifications."""

    @property
    def policy_id(self) -> str:
        return "governance.high_risk_approval"

    def evaluate(self, request: ToolAuthorizationRequest) -> PolicyEvaluationResult:
        if request.risk_level in (
            ToolAuthorizationRiskLevel.HIGH,
            ToolAuthorizationRiskLevel.CRITICAL,
        ):
            if request.approval_evidence_ref:
                return PolicyEvaluationResult(
                    policy_id=self.policy_id,
                    decision=ToolAuthorizationDecisionState.ALLOW,
                    reason="high_risk_approval_evidence_present",
                )
            return PolicyEvaluationResult(
                policy_id=self.policy_id,
                decision=ToolAuthorizationDecisionState.REQUIRE_APPROVAL,
                reason=f"high_risk_requires_approval:{request.risk_level.value}",
            )
        return PolicyEvaluationResult(
            policy_id=self.policy_id,
            decision=ToolAuthorizationDecisionState.ALLOW,
            reason="risk_below_approval_threshold",
        )
