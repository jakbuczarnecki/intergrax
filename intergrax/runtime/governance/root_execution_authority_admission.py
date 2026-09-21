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
)
from intergrax.contracts.evaluated_policy_decision import request_digest_for_payload
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernedExecutionEvaluationPoint,
    build_governance_fact_from_policy_decision,
    is_canonical_governance_evidence_policy_action,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder
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


def _approved_scopes_exceed_collaborative_authority(
    collaborative_scopes: tuple[str, ...],
    runtime_approved_scopes: tuple[str, ...] | None,
) -> bool:
    if runtime_approved_scopes is None:
        return False
    collaborative = set(collaborative_scopes)
    return not set(runtime_approved_scopes).issubset(collaborative)


class RootExecutionAuthorityAdmissionService:
    """Runtime admission — collaborative ALLOW is necessary but not sufficient."""

    def __init__(
        self,
        *,
        runtime_policy_admission: RuntimeExecutionPolicyAdmissionPort,
        governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
    ) -> None:
        self._runtime_policy_admission = runtime_policy_admission
        self._governance_evidence_recorder = governance_evidence_recorder

    def _record_evidence(
        self,
        request: RootExecutionAuthorityAdmissionRequest,
        decision: PolicyDecision,
    ) -> None:
        recorder = self._governance_evidence_recorder
        if recorder is None or recorder.persistence is None:
            return
        if not is_canonical_governance_evidence_policy_action(decision.action):
            return
        digest = request_digest_for_payload(
            {
                "tenant_id": request.tenant_id,
                "workspace_id": request.workspace_id,
                "principal_id": request.principal_id,
                "root_execution_operation": request.root_execution_operation.value,
                "collaborative_scopes": request.collaborative_authority_scopes,
            }
        )
        idempotency_key = f"root_admission:{digest}:{decision.action.value}:{decision.policy_rule_id}"
        fact = build_governance_fact_from_policy_decision(
            evaluation_point=GovernedExecutionEvaluationPoint.ROOT_EXECUTION_ADMISSION,
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            principal_id=request.principal_id,
            decision=decision,
            request_digest=digest,
            idempotency_key=idempotency_key,
            action=request.root_execution_operation.policy_operation(),
            resource_type="root_execution",
            resource_scope=request.root_execution_operation.value,
            task_id=request.task_id,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            execution_id=request.execution_id,
        )
        recorder.record(fact)

    def authorize(
        self,
        request: RootExecutionAuthorityAdmissionRequest,
    ) -> RootExecutionAuthorityAdmissionResult:
        collaborative_decision = request.effective_authority_decision.decision
        if collaborative_decision.action is not PolicyAction.ALLOW:
            result = RootExecutionAuthorityAdmissionResult(
                disposition=_map_collaborative_evidence(collaborative_decision.action),
                policy_decision=collaborative_decision,
            )
            self._record_evidence(request, collaborative_decision)
            return result

        runtime_result = self._runtime_policy_admission.evaluate(
            RuntimeExecutionPolicyAdmissionRequest(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                principal_id=request.principal_id,
                collaborative_authority_scopes=request.collaborative_authority_scopes,
                execution_operation=request.root_execution_operation.policy_operation(),
            )
        )
        if _approved_scopes_exceed_collaborative_authority(
            request.collaborative_authority_scopes,
            runtime_result.approved_scopes,
        ):
            deny = PolicyDecision(
                action=PolicyAction.DENY,
                reason="runtime_approved_scopes_exceed_collaborative_authority",
                policy_rule_id="runtime.root_execution_admission.scope_widening",
            )
            result = RootExecutionAuthorityAdmissionResult(
                disposition=RootExecutionAuthorityAdmissionDisposition.DENIED,
                policy_decision=deny,
            )
            self._record_evidence(request, deny)
            return result
        runtime_decision = runtime_result.policy_decision
        disposition = _map_runtime_policy_action(runtime_decision)
        if disposition is not RootExecutionAuthorityAdmissionDisposition.ALLOWED:
            result = RootExecutionAuthorityAdmissionResult(
                disposition=disposition,
                policy_decision=runtime_decision,
            )
            self._record_evidence(request, runtime_decision)
            return result

        trusted_scopes = _narrow_collaborative_scopes(
            request.collaborative_authority_scopes,
            runtime_result.approved_scopes,
        )
        if not trusted_scopes:
            deny = PolicyDecision(
                action=PolicyAction.DENY,
                reason="runtime_approved_scopes_empty_after_narrowing",
                policy_rule_id="runtime.root_execution_admission.scope_narrowing",
            )
            result = RootExecutionAuthorityAdmissionResult(
                disposition=RootExecutionAuthorityAdmissionDisposition.DENIED,
                policy_decision=deny,
            )
            self._record_evidence(request, deny)
            return result

        trusted = ParentExecutionAuthority.scoped(trusted_scopes)
        result = RootExecutionAuthorityAdmissionResult(
            disposition=RootExecutionAuthorityAdmissionDisposition.ALLOWED,
            trusted_parent_execution_authority=trusted,
            policy_decision=runtime_decision,
        )
        self._record_evidence(request, runtime_decision)
        return result


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
