# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime execution policy admission adapters (AW-5A corrective)."""

from __future__ import annotations

from intergrax.contracts.runtime_execution_policy_admission import (
    RuntimeExecutionPolicyAdmissionPort,
    RuntimeExecutionPolicyAdmissionRequest,
    RuntimeExecutionPolicyAdmissionResult,
)
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine


class RuntimeExecutionPolicyAdmissionEvaluator(RuntimeExecutionPolicyAdmissionPort):
    """Canonical adapter over ``RuntimePolicyEngine`` root execution admission."""

    def __init__(self, *, policy_engine: RuntimePolicyEngine) -> None:
        self._policy_engine = policy_engine

    def evaluate(
        self,
        request: RuntimeExecutionPolicyAdmissionRequest,
    ) -> RuntimeExecutionPolicyAdmissionResult:
        decision, approved_scopes = self._policy_engine.evaluate_root_execution_admission(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            principal_id=request.principal_id,
            collaborative_authority_scopes=request.collaborative_authority_scopes,
            execution_operation=request.execution_operation,
            resource_scope=request.resource_scope,
        )
        return RuntimeExecutionPolicyAdmissionResult(
            policy_decision=decision,
            approved_scopes=approved_scopes,
        )


class AllowingRuntimeExecutionPolicyAdmission(RuntimeExecutionPolicyAdmissionPort):
    """Test/reference adapter — runtime ALLOW without scope narrowing."""

    def evaluate(
        self,
        request: RuntimeExecutionPolicyAdmissionRequest,
    ) -> RuntimeExecutionPolicyAdmissionResult:
        del request
        return RuntimeExecutionPolicyAdmissionResult(
            policy_decision=PolicyDecision(
                action=PolicyAction.ALLOW,
                reason="runtime_execution_policy_allow",
                policy_rule_id="test.runtime_execution.allow",
            ),
        )


class DenyingRuntimeExecutionPolicyAdmission(RuntimeExecutionPolicyAdmissionPort):
    """Test/reference adapter — runtime DENY."""

    def evaluate(
        self,
        request: RuntimeExecutionPolicyAdmissionRequest,
    ) -> RuntimeExecutionPolicyAdmissionResult:
        del request
        return RuntimeExecutionPolicyAdmissionResult(
            policy_decision=PolicyDecision(
                action=PolicyAction.DENY,
                reason="runtime_execution_policy_denied",
                policy_rule_id="test.runtime_execution.deny",
            ),
        )


class RequireHumanRuntimeExecutionPolicyAdmission(RuntimeExecutionPolicyAdmissionPort):
    """Test/reference adapter — runtime REQUIRE_HUMAN."""

    def evaluate(
        self,
        request: RuntimeExecutionPolicyAdmissionRequest,
    ) -> RuntimeExecutionPolicyAdmissionResult:
        del request
        return RuntimeExecutionPolicyAdmissionResult(
            policy_decision=PolicyDecision(
                action=PolicyAction.REQUIRE_HUMAN,
                reason="runtime_execution_policy_require_human",
                policy_rule_id="test.runtime_execution.require_human",
            ),
        )


class EscalateRuntimeExecutionPolicyAdmission(RuntimeExecutionPolicyAdmissionPort):
    """Test/reference adapter — runtime ESCALATE."""

    def evaluate(
        self,
        request: RuntimeExecutionPolicyAdmissionRequest,
    ) -> RuntimeExecutionPolicyAdmissionResult:
        del request
        return RuntimeExecutionPolicyAdmissionResult(
            policy_decision=PolicyDecision(
                action=PolicyAction.ESCALATE,
                reason="runtime_execution_policy_escalate",
                policy_rule_id="test.runtime_execution.escalate",
            ),
        )


class UnavailableRuntimeExecutionPolicyAdmission(RuntimeExecutionPolicyAdmissionPort):
    """Fail-closed adapter when runtime policy evaluation is unavailable."""

    def evaluate(
        self,
        request: RuntimeExecutionPolicyAdmissionRequest,
    ) -> RuntimeExecutionPolicyAdmissionResult:
        del request
        return RuntimeExecutionPolicyAdmissionResult(
            policy_decision=PolicyDecision(
                action=PolicyAction.DENY,
                reason="runtime_execution_policy_unavailable",
                enforcement_level=EnforcementLevel.MANDATORY,
                policy_rule_id="runtime.execution_policy.unavailable",
            ),
        )
