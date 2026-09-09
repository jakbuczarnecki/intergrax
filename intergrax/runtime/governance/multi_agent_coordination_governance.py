# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical pre-coordination governance boundary (NPSC-5D/R1 / R1-H1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.autonomous_work.execution_authority_admission import (
    CollaborativeWorkAuthorityResolverPort,
)
from intergrax.collaborative_work.policy_composition import compose_policy_decisions
from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    DelegationStatus,
    EffectiveAuthorityRequest,
    PolicyCompositionApplicability,
    PolicyCompositionInput,
    PolicyLayerApplicability,
)
from intergrax.contracts.multi_agent_coordination_governance import (
    MULTI_AGENT_COORDINATION_COLLABORATIVE_AUTHORITY_SCOPE,
    MultiAgentCoordinationCollaborativeApplicability,
    MultiAgentCoordinationGovernanceEvaluator,
    MultiAgentCoordinationGovernancePort,
    MultiAgentCoordinationGovernanceRequest,
    MultiAgentCoordinationGovernanceResult,
    evidence_from_request_and_decision,
    multi_agent_coordination_governance_request_digest,
)
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision


@runtime_checkable
class MultiAgentCoordinationPolicyEvaluatorPort(Protocol):
    """Canonical coordination policy evaluation surface."""

    def evaluate_multi_agent_coordination(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> PolicyDecision:
        """Evaluate semantic coordination admission policy for ``request``."""
        ...


class MultiAgentCoordinationGovernanceBoundary:
    """Shared authorization boundary for MULTI_AGENT_COORDINATION evaluation."""

    def __init__(
        self,
        *,
        evaluator: MultiAgentCoordinationGovernanceEvaluator,
        authority_resolver: CollaborativeWorkAuthorityResolverPort | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._authority_resolver = authority_resolver

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

        coordination_decision = self._evaluator.evaluate(request)
        decision = self._compose_with_collaborative_authority(
            request,
            coordination_decision=coordination_decision,
        )
        if decision is None:
            return self._fail_closed(
                request,
                reason="collaborative_authority_resolver_unavailable",
                validation_failed=False,
            )

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

    def _compose_with_collaborative_authority(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
        *,
        coordination_decision: PolicyDecision,
    ) -> PolicyDecision | None:
        if (
            request.collaborative_applicability
            is MultiAgentCoordinationCollaborativeApplicability.NOT_APPLICABLE
        ):
            return coordination_decision

        if self._authority_resolver is None:
            return None

        collaborative_context = request.collaborative_context
        if collaborative_context is None:
            return None

        authority_request = EffectiveAuthorityRequest(
            tenant_id=request.tenant_id,
            workspace_id=collaborative_context.workspace_id,
            acting_principal_id=collaborative_context.acting_principal_id,
            requested_authority_scopes=(
                MULTI_AGENT_COORDINATION_COLLABORATIVE_AUTHORITY_SCOPE,
            ),
            delegator_principal_id=collaborative_context.delegator_principal_id,
            resource_scope=collaborative_context.resource_scope,
            delegation=self._delegation_locator(request, collaborative_context),
            membership_resolution_mode=collaborative_context.membership_resolution_mode,
        )
        authority_decision = self._authority_resolver.resolve(authority_request)
        composed = compose_policy_decisions(
            PolicyCompositionInput(
                collaborative_authority=authority_decision.decision,
                runtime_policy=coordination_decision,
                applicability=PolicyCompositionApplicability(
                    workspace_policy=PolicyLayerApplicability.NOT_APPLICABLE,
                    resource_policy=PolicyLayerApplicability.NOT_APPLICABLE,
                    runtime_policy=PolicyLayerApplicability.REQUIRED,
                ),
            ),
        )
        return composed.decision

    @staticmethod
    def _delegation_locator(
        request: MultiAgentCoordinationGovernanceRequest,
        collaborative_context,
    ) -> AuthorityDelegation | None:
        if collaborative_context.delegation_id is None:
            return None
        if collaborative_context.delegator_principal_id is None:
            return None
        return AuthorityDelegation(
            delegation_id=collaborative_context.delegation_id,
            tenant_id=request.tenant_id,
            workspace_id=collaborative_context.workspace_id,
            delegator_principal_id=collaborative_context.delegator_principal_id,
            delegate_principal_id=collaborative_context.acting_principal_id,
            authority_scopes=(MULTI_AGENT_COORDINATION_COLLABORATIVE_AUTHORITY_SCOPE,),
            status=DelegationStatus.ACTIVE,
            revision=0,
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
        if (
            request.collaborative_applicability
            is MultiAgentCoordinationCollaborativeApplicability.REQUIRED
            and request.collaborative_context is None
        ):
            return "missing_collaborative_context"
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

    def __init__(
        self,
        *,
        authority_resolver: CollaborativeWorkAuthorityResolverPort | None = None,
    ) -> None:
        self._authority_resolver = authority_resolver

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
            authority_resolver=self._authority_resolver,
        )
        return boundary.evaluate(request)


class DenyingMultiAgentCoordinationGovernance:
    """Test/reference adapter — coordination DENY."""

    def __init__(
        self,
        *,
        authority_resolver: CollaborativeWorkAuthorityResolverPort | None = None,
    ) -> None:
        self._authority_resolver = authority_resolver

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
            authority_resolver=self._authority_resolver,
        )
        return boundary.evaluate(request)


class RequireHumanMultiAgentCoordinationGovernance:
    """Test/reference adapter — coordination REQUIRE_HUMAN."""

    def __init__(
        self,
        *,
        authority_resolver: CollaborativeWorkAuthorityResolverPort | None = None,
    ) -> None:
        self._authority_resolver = authority_resolver

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
            authority_resolver=self._authority_resolver,
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
    """Canonical adapter over coordination policy evaluator."""

    def __init__(self, *, policy_engine: MultiAgentCoordinationPolicyEvaluatorPort) -> None:
        if not isinstance(policy_engine, MultiAgentCoordinationPolicyEvaluatorPort):
            raise TypeError(
                "policy_engine must implement MultiAgentCoordinationPolicyEvaluatorPort",
            )
        self._policy_engine = policy_engine

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> PolicyDecision:
        return self._policy_engine.evaluate_multi_agent_coordination(request)


class RuntimeMultiAgentCoordinationGovernance:
    """Port adapter composing boundary + coordination policy evaluator."""

    def __init__(
        self,
        *,
        policy_engine: MultiAgentCoordinationPolicyEvaluatorPort,
        authority_resolver: CollaborativeWorkAuthorityResolverPort | None = None,
    ) -> None:
        self._evaluator = RuntimeMultiAgentCoordinationGovernanceEvaluator(
            policy_engine=policy_engine,
        )
        self._boundary = MultiAgentCoordinationGovernanceBoundary(
            evaluator=self._evaluator,
            authority_resolver=authority_resolver,
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

    def __init__(self, *, policy_engine: MultiAgentCoordinationPolicyEvaluatorPort) -> None:
        self._policy_engine = policy_engine

    def evaluate(
        self,
        request: MultiAgentCoordinationGovernanceRequest,
    ) -> PolicyDecision:
        return self._policy_engine.evaluate_multi_agent_coordination(request)


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
