# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime/Governance trusted root execution authority admission (AW-5A seam).

Mints trusted ``ParentExecutionAuthority`` for canonical root execution intake.
Autonomous Work must consume this port — it must not mint trusted authority.
"""

from __future__ import annotations

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.runtime_execution_admission import (
    RootExecutionAuthorityAdmissionDisposition,
    RootExecutionAuthorityAdmissionRequest,
    RootExecutionAuthorityAdmissionResult,
)
from intergrax.contracts.runtime_execution_policy_admission import (
    RuntimeExecutionPolicyAdmissionPort,
    RuntimeExecutionPolicyAdmissionRequest,
    WORKER_ROOT_EXECUTION_OPERATION,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    RuntimeExecutionPolicyAdmissionEvaluator,
    default_worker_root_execution_policy_engine,
)


def _map_runtime_policy_action(
    decision: PolicyDecision,
) -> RootExecutionAuthorityAdmissionDisposition:
    action = decision.action
    if action is PolicyAction.ALLOW:
        return RootExecutionAuthorityAdmissionDisposition.ALLOWED
    if action is PolicyAction.DENY:
        if decision.reason in {
            "root_execution_admission_unconfigured",
            "runtime_execution_policy_unavailable",
        }:
            return RootExecutionAuthorityAdmissionDisposition.UNAVAILABLE
        return RootExecutionAuthorityAdmissionDisposition.DENIED
    if action is PolicyAction.REQUIRE_HUMAN:
        return RootExecutionAuthorityAdmissionDisposition.REQUIRE_HUMAN
    if action is PolicyAction.ESCALATE:
        return RootExecutionAuthorityAdmissionDisposition.ESCALATE
    if action is PolicyAction.MODIFY:
        return RootExecutionAuthorityAdmissionDisposition.DENIED
    return RootExecutionAuthorityAdmissionDisposition.UNAVAILABLE


def _narrow_collaborative_scopes(
    collaborative_scopes: tuple[str, ...],
    runtime_approved_scopes: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if runtime_approved_scopes is None:
        return collaborative_scopes
    approved = set(runtime_approved_scopes)
    return tuple(scope for scope in collaborative_scopes if scope in approved)


class RootExecutionAuthorityAdmissionService:
    """Runtime admission — collaborative ALLOW is necessary but not sufficient."""

    def __init__(
        self,
        *,
        runtime_policy_admission: RuntimeExecutionPolicyAdmissionPort | None = None,
    ) -> None:
        if runtime_policy_admission is None:
            runtime_policy_admission = RuntimeExecutionPolicyAdmissionEvaluator(
                policy_engine=default_worker_root_execution_policy_engine(),
            )
        self._runtime_policy_admission = runtime_policy_admission

    def authorize(
        self,
        request: RootExecutionAuthorityAdmissionRequest,
    ) -> RootExecutionAuthorityAdmissionResult:
        collaborative_decision = request.effective_authority_decision.decision
        if collaborative_decision.action is not PolicyAction.ALLOW:
            return RootExecutionAuthorityAdmissionResult(
                disposition=_map_collaborative_evidence(collaborative_decision.action),
                policy_decision=collaborative_decision,
            )

        runtime_result = self._runtime_policy_admission.evaluate(
            RuntimeExecutionPolicyAdmissionRequest(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                principal_id=request.principal_id,
                collaborative_authority_scopes=request.collaborative_authority_scopes,
                execution_operation=WORKER_ROOT_EXECUTION_OPERATION,
            )
        )
        runtime_decision = runtime_result.policy_decision
        disposition = _map_runtime_policy_action(runtime_decision)
        if disposition is not RootExecutionAuthorityAdmissionDisposition.ALLOWED:
            return RootExecutionAuthorityAdmissionResult(
                disposition=disposition,
                policy_decision=runtime_decision,
            )

        trusted_scopes = _narrow_collaborative_scopes(
            request.collaborative_authority_scopes,
            runtime_result.approved_scopes,
        )
        if not trusted_scopes:
            return RootExecutionAuthorityAdmissionResult(
                disposition=RootExecutionAuthorityAdmissionDisposition.DENIED,
                policy_decision=PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="runtime_approved_scopes_empty_after_narrowing",
                    policy_rule_id="runtime.root_execution_admission.scope_narrowing",
                ),
            )

        trusted = ParentExecutionAuthority.scoped(trusted_scopes)
        return RootExecutionAuthorityAdmissionResult(
            disposition=RootExecutionAuthorityAdmissionDisposition.ALLOWED,
            trusted_parent_execution_authority=trusted,
            policy_decision=runtime_decision,
        )


def _map_collaborative_evidence(
    action: PolicyAction,
) -> RootExecutionAuthorityAdmissionDisposition:
    if action is PolicyAction.DENY:
        return RootExecutionAuthorityAdmissionDisposition.DENIED
    if action is PolicyAction.REQUIRE_HUMAN:
        return RootExecutionAuthorityAdmissionDisposition.REQUIRE_HUMAN
    if action is PolicyAction.ESCALATE:
        return RootExecutionAuthorityAdmissionDisposition.ESCALATE
    if action is PolicyAction.MODIFY:
        return RootExecutionAuthorityAdmissionDisposition.DENIED
    return RootExecutionAuthorityAdmissionDisposition.UNAVAILABLE


class DenyingRootExecutionAuthorityAdmission:
    """Test/reference adapter that always denies runtime admission."""

    def __init__(
        self,
        *,
        runtime_policy_admission: RuntimeExecutionPolicyAdmissionPort | None = None,
    ) -> None:
        from intergrax.runtime.governance.runtime_execution_policy_admission import (
            DenyingRuntimeExecutionPolicyAdmission,
        )

        self._service = RootExecutionAuthorityAdmissionService(
            runtime_policy_admission=(
                runtime_policy_admission or DenyingRuntimeExecutionPolicyAdmission()
            ),
        )

    def authorize(
        self,
        request: RootExecutionAuthorityAdmissionRequest,
    ) -> RootExecutionAuthorityAdmissionResult:
        return self._service.authorize(request)


class UnavailableRootExecutionAuthorityAdmission:
    """Fail-closed adapter when runtime admission is unavailable."""

    def __init__(
        self,
        *,
        runtime_policy_admission: RuntimeExecutionPolicyAdmissionPort | None = None,
    ) -> None:
        from intergrax.runtime.governance.runtime_execution_policy_admission import (
            UnavailableRuntimeExecutionPolicyAdmission,
        )

        self._service = RootExecutionAuthorityAdmissionService(
            runtime_policy_admission=(
                runtime_policy_admission or UnavailableRuntimeExecutionPolicyAdmission()
            ),
        )

    def authorize(
        self,
        request: RootExecutionAuthorityAdmissionRequest,
    ) -> RootExecutionAuthorityAdmissionResult:
        return self._service.authorize(request)
