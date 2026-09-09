# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical post-selection physical delegation governance boundary (NPSC-5D/R2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.physical_delegation_governance import (
    PhysicalDelegationGovernanceEvaluator,
    PhysicalDelegationGovernancePort,
    PhysicalDelegationGovernanceRequest,
    PhysicalDelegationGovernanceResult,
    evidence_from_request_and_decision,
    physical_delegation_governance_request_digest,
)
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision


@runtime_checkable
class PhysicalDelegationPolicyEvaluatorPort(Protocol):
    """Canonical physical delegation policy evaluation surface."""

    def evaluate_physical_delegation(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PolicyDecision:
        """Evaluate physical delegation admission policy for ``request``."""
        ...


class PhysicalDelegationGovernanceBoundary:
    """Shared authorization boundary for MULTI_AGENT_DELEGATION evaluation."""

    def __init__(
        self,
        *,
        evaluator: PhysicalDelegationGovernanceEvaluator,
    ) -> None:
        self._evaluator = evaluator

    @property
    def evaluator(self) -> PhysicalDelegationGovernanceEvaluator:
        return self._evaluator

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
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
                reason="modify_not_supported_for_physical_delegation",
                decision=decision,
                validation_failed=False,
            )

        request_digest = physical_delegation_governance_request_digest(request)
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
        return PhysicalDelegationGovernanceResult(
            permitted=permitted,
            decision=decision,
            evidence=evidence,
            requires_governed_continuation=requires_continuation,
            validation_failed=False,
        )

    @staticmethod
    def _validate_request(
        request: PhysicalDelegationGovernanceRequest,
    ) -> str | None:
        principal = request.principal
        if not principal.tenant_id.strip():
            return "missing_tenant"
        if not (principal.user_id or principal.auth_subject):
            return "missing_principal_identity"
        if not str(request.delegation_id).strip():
            return "missing_delegation_identity"
        if not str(request.task_scope_id).strip():
            return "missing_task_scope"
        if not request.application_id.strip() or not request.application_environment_id.strip():
            return "missing_application_scope"
        if not request.capability_requirement.required_capability_ids:
            return "missing_capability_requirement"
        return None

    def _fail_closed(
        self,
        request: PhysicalDelegationGovernanceRequest,
        *,
        reason: str,
        decision: PolicyDecision | None = None,
        validation_failed: bool,
    ) -> PhysicalDelegationGovernanceResult:
        resolved = decision or PolicyDecision(
            action=PolicyAction.DENY,
            reason=reason,
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="physical_delegation.governance.fail_closed",
        )
        request_digest = physical_delegation_governance_request_digest(request)
        evidence = evidence_from_request_and_decision(
            request,
            decision=resolved,
            request_digest=request_digest,
        )
        return PhysicalDelegationGovernanceResult(
            permitted=False,
            decision=resolved,
            evidence=evidence,
            requires_governed_continuation=False,
            validation_failed=validation_failed,
        )


class AllowingPhysicalDelegationGovernance:
    """Test/reference adapter — physical delegation ALLOW without authority widening."""

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        boundary = PhysicalDelegationGovernanceBoundary(
            evaluator=_StaticPhysicalDelegationGovernanceEvaluator(
                PolicyDecision(
                    action=PolicyAction.ALLOW,
                    reason="physical_delegation_allow",
                    policy_rule_id="test.physical_delegation.allow",
                ),
            ),
        )
        return boundary.evaluate(request)


class DenyingPhysicalDelegationGovernance:
    """Test/reference adapter — physical delegation DENY."""

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        boundary = PhysicalDelegationGovernanceBoundary(
            evaluator=_StaticPhysicalDelegationGovernanceEvaluator(
                PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="physical_delegation_denied",
                    policy_rule_id="test.physical_delegation.deny",
                ),
            ),
        )
        return boundary.evaluate(request)


class RequireHumanPhysicalDelegationGovernance:
    """Test/reference adapter — physical delegation REQUIRE_HUMAN."""

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        boundary = PhysicalDelegationGovernanceBoundary(
            evaluator=_StaticPhysicalDelegationGovernanceEvaluator(
                PolicyDecision(
                    action=PolicyAction.REQUIRE_HUMAN,
                    reason="physical_delegation_require_human",
                    policy_rule_id="test.physical_delegation.require_human",
                ),
            ),
        )
        return boundary.evaluate(request)


class UnavailablePhysicalDelegationGovernance:
    """Fail-closed adapter when physical delegation governance is unavailable."""

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        boundary = PhysicalDelegationGovernanceBoundary(
            evaluator=_UnavailablePhysicalDelegationGovernanceEvaluator(),
        )
        return boundary.evaluate(request)


class RuntimePhysicalDelegationGovernanceEvaluator:
    """Canonical adapter over physical delegation policy evaluator."""

    def __init__(self, *, policy_engine: PhysicalDelegationPolicyEvaluatorPort) -> None:
        if not isinstance(policy_engine, PhysicalDelegationPolicyEvaluatorPort):
            raise TypeError(
                "policy_engine must implement PhysicalDelegationPolicyEvaluatorPort",
            )
        self._policy_engine = policy_engine

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PolicyDecision:
        return self._policy_engine.evaluate_physical_delegation(request)


class RuntimePhysicalDelegationGovernance:
    """Port adapter composing boundary + physical delegation policy evaluator."""

    def __init__(
        self,
        *,
        policy_engine: PhysicalDelegationPolicyEvaluatorPort,
    ) -> None:
        self._evaluator = RuntimePhysicalDelegationGovernanceEvaluator(
            policy_engine=policy_engine,
        )
        self._boundary = PhysicalDelegationGovernanceBoundary(
            evaluator=self._evaluator,
        )

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        return self._boundary.evaluate(request)


class _StaticPhysicalDelegationGovernanceEvaluator:
    def __init__(self, decision: PolicyDecision) -> None:
        self._decision = decision

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PolicyDecision:
        del request
        return self._decision


class _UnavailablePhysicalDelegationGovernanceEvaluator:
    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PolicyDecision:
        del request
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="physical_delegation_governance_unavailable",
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="physical_delegation.governance.unavailable",
        )


def as_physical_delegation_governance_port(
    port: PhysicalDelegationGovernancePort | PhysicalDelegationGovernanceBoundary,
) -> PhysicalDelegationGovernancePort:
    if isinstance(port, PhysicalDelegationGovernanceBoundary):
        return _BoundaryPortAdapter(port)
    return port


class _BoundaryPortAdapter:
    __slots__ = ("_boundary",)

    def __init__(self, boundary: PhysicalDelegationGovernanceBoundary) -> None:
        self._boundary = boundary

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        return self._boundary.evaluate(request)
