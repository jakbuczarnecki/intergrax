# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical pre-coordination governance boundary (NPSC-5D/R1)."""

from __future__ import annotations

from intergrax.contracts.multi_agent_coordination_governance import (
    MultiAgentCoordinationGovernanceEvaluator,
    MultiAgentCoordinationGovernancePort,
    MultiAgentCoordinationGovernanceRequest,
    MultiAgentCoordinationGovernanceResult,
    evidence_from_request_and_decision,
    multi_agent_coordination_governance_request_digest,
)
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision


class MultiAgentCoordinationGovernanceBoundary:
    """Shared authorization boundary for MULTI_AGENT_COORDINATION evaluation."""

    def __init__(self, *, evaluator: MultiAgentCoordinationGovernanceEvaluator) -> None:
        self._evaluator = evaluator

    @property
    def evaluator(self) -> MultiAgentCoordinationGovernanceEvaluator:
        return self._evaluator

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> MultiAgentCoordinationGovernanceResult:
        validation_reason = self._validate_request(request)
        if validation_reason is not None:
            return self._fail_closed(
                request,
                reason=validation_reason,
                validation_failed=True,
            )

        decision = self._evaluator.evaluate(request)

        if decision.action is PolicyAction.MODIFY:
            return self._fail_closed(
                request,
                reason="modify_not_supported_for_multi_agent_coordination",
                decision=decision,
                validation_failed=False,
            )

        request_digest = multi_agent_coordination_governance_request_digest(request)
        evidence = evidence_from_request_and_decision(
            request,
            decision=decision,
            request_digest=request_digest,
        )
        action = decision.action
        permitted = action is PolicyAction.ALLOW
        requires_continuation = action in (
            PolicyAction.REQUIRE_HUMAN,
            PolicyAction.ESCALATE,
        )
        return MultiAgentCoordinationGovernanceResult(
            permitted=permitted,
            decision=decision,
            evidence=evidence,
            requires_governed_continuation=requires_continuation,
            validation_failed=False,
        )

    @staticmethod
    def _validate_request(
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> str | None:
        principal = request.principal
        if not principal.tenant_id.strip():
            return "missing_tenant"
        if not (principal.user_id or principal.auth_subject):
            return "missing_principal_identity"
        if not request.intent_id.strip():
            return "missing_intent_identity"
        if not request.task_scope_id.strip():
            return "missing_task_scope"
        if not request.application_id.strip() or not request.application_environment_id.strip():
            return "missing_application_scope"
        if not request.contributions:
            return "missing_contributions"
        return None

    def _fail_closed(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
        *,
        reason: str,
        decision: PolicyDecision | None = None,
        validation_failed: bool,
    ) -> MultiAgentCoordinationGovernanceResult:
        resolved = decision or PolicyDecision(
            action=PolicyAction.DENY,
            reason=reason,
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="multi_agent_coordination.governance.fail_closed",
        )
        request_digest = multi_agent_coordination_governance_request_digest(request)
        evidence = evidence_from_request_and_decision(
            request,
            decision=resolved,
            request_digest=request_digest,
        )
        return MultiAgentCoordinationGovernanceResult(
            permitted=False,
            decision=resolved,
            evidence=evidence,
            requires_governed_continuation=False,
            validation_failed=validation_failed,
        )


class AllowingMultiAgentCoordinationGovernance:
    """Test/reference adapter — coordination ALLOW without authority widening."""

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> MultiAgentCoordinationGovernanceResult:
        boundary = MultiAgentCoordinationGovernanceBoundary(
            evaluator=_StaticMultiAgentCoordinationGovernanceEvaluator(
                PolicyDecision(
                    action=PolicyAction.ALLOW,
                    reason="multi_agent_coordination_allow",
                    policy_rule_id="test.multi_agent_coordination.allow",
                ),
            ),
        )
        return boundary.evaluate(request)


class DenyingMultiAgentCoordinationGovernance:
    """Test/reference adapter — coordination DENY."""

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> MultiAgentCoordinationGovernanceResult:
        boundary = MultiAgentCoordinationGovernanceBoundary(
            evaluator=_StaticMultiAgentCoordinationGovernanceEvaluator(
                PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="multi_agent_coordination_denied",
                    policy_rule_id="test.multi_agent_coordination.deny",
                ),
            ),
        )
        return boundary.evaluate(request)


class RequireHumanMultiAgentCoordinationGovernance:
    """Test/reference adapter — coordination REQUIRE_HUMAN."""

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> MultiAgentCoordinationGovernanceResult:
        boundary = MultiAgentCoordinationGovernanceBoundary(
            evaluator=_StaticMultiAgentCoordinationGovernanceEvaluator(
                PolicyDecision(
                    action=PolicyAction.REQUIRE_HUMAN,
                    reason="multi_agent_coordination_require_human",
                    policy_rule_id="test.multi_agent_coordination.require_human",
                ),
            ),
        )
        return boundary.evaluate(request)


class UnavailableMultiAgentCoordinationGovernance:
    """Fail-closed adapter when coordination governance evaluation is unavailable."""

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> MultiAgentCoordinationGovernanceResult:
        boundary = MultiAgentCoordinationGovernanceBoundary(
            evaluator=_UnavailableMultiAgentCoordinationGovernanceEvaluator(),
        )
        return boundary.evaluate(request)


class RuntimeMultiAgentCoordinationGovernanceEvaluator:
    """Canonical adapter over ``RuntimePolicyEngine`` coordination admission."""

    def __init__(self, *, policy_engine) -> None:
        from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

        if not isinstance(policy_engine, RuntimePolicyEngine):
            raise TypeError("policy_engine must be RuntimePolicyEngine")
        self._policy_engine = policy_engine

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> PolicyDecision:
        return self._policy_engine.evaluate_multi_agent_coordination(request)


class RuntimeMultiAgentCoordinationGovernance:
    """Port adapter composing boundary + runtime policy engine."""

    def __init__(self, *, policy_engine) -> None:
        self._evaluator = RuntimeMultiAgentCoordinationGovernanceEvaluator(
            policy_engine=policy_engine,
        )
        self._boundary = MultiAgentCoordinationGovernanceBoundary(
            evaluator=self._evaluator,
        )

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> MultiAgentCoordinationGovernanceResult:
        return self._boundary.evaluate(request)


class _StaticMultiAgentCoordinationGovernanceEvaluator:
    def __init__(self, decision: PolicyDecision) -> None:
        self._decision = decision

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> PolicyDecision:
        del request
        return self._decision


class _UnavailableMultiAgentCoordinationGovernanceEvaluator:
    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> PolicyDecision:
        del request
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="multi_agent_coordination_governance_unavailable",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="multi_agent_coordination.governance.unavailable",
        )


class FailClosedMultiAgentCoordinationGovernanceEvaluator:
    """Evaluator that maps indeterminate policy outcomes to DENY."""

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> PolicyDecision:
        from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

        return RuntimePolicyEngine().evaluate_multi_agent_coordination(request)


def as_coordination_governance_port(
    port: MultiAgentCoordinationGovernancePort | MultiAgentCoordinationGovernanceBoundary,
) -> MultiAgentCoordinationGovernancePort:
    if isinstance(port, MultiAgentCoordinationGovernanceBoundary):
        return _BoundaryPortAdapter(port)
    return port


class _BoundaryPortAdapter:
    __slots__ = ("_boundary",)

    def __init__(self, boundary: MultiAgentCoordinationGovernanceBoundary) -> None:
        self._boundary = boundary

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> MultiAgentCoordinationGovernanceResult:
        return self._boundary.evaluate(request)
