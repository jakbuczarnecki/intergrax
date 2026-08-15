# © Artur Czarnecki. All rights reserved.

"""Trusted operation classification and final enforcement gate (COLLAB-WORK-1G).

Loads authoritative operation policy profiles, resolves collaborative authority
using profile-owned scope, evaluates required policy layers, and composes a final
``PolicyDecision``. Does not execute operations or accept caller-controlled
applicability.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.policy_composition import compose_policy_decisions
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import CollaborativeOperationPolicyProfileRepository
from intergrax.contracts.collaborative_work import (
    CollaborativeOperationPolicyProfile,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    EffectiveAuthorityRequest,
    OperationPolicyRequirement,
    PolicyCompositionApplicability,
    PolicyCompositionInput,
    PolicyCompositionResult,
    PolicyLayerApplicability,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

_POLICY_RULE_PROFILE_MISSING = "collaborative_work.operation_profile.missing"
_POLICY_RULE_PROFILE_INACTIVE = "collaborative_work.operation_profile.inactive"
_POLICY_RULE_RESOURCE_REQUIRED_MISSING = "collaborative_work.enforcement.resource_required_missing"
_POLICY_RULE_RUNTIME_REQUEST_MISSING = "collaborative_work.enforcement.runtime_request_missing"
_POLICY_RULE_RUNTIME_IDENTITY_MISMATCH = "collaborative_work.enforcement.runtime_identity_mismatch"


@runtime_checkable
class MeaningfulSideEffectPolicyEvaluator(Protocol):
    """Runtime meaningful-side-effect evaluation surface reused by the gate."""

    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        """Evaluate a proposed external side effect — fail closed by default."""


def _classification_deny(reason: str, policy_rule_id: str) -> PolicyDecision:
    return PolicyDecision(
        action=PolicyAction.DENY,
        reason=reason,
        policy_rule_id=policy_rule_id,
    )


def _classification_result(
    *,
    operation_id: str,
    reason: str,
    policy_rule_id: str,
) -> CollaborativeWorkEnforcementResult:
    deny = _classification_deny(reason, policy_rule_id)
    return CollaborativeWorkEnforcementResult(
        operation_id=operation_id,
        profile_revision=None,
        authority_scope=None,
        composition=PolicyCompositionResult(
            decision=deny,
            collaborative_authority=deny,
        ),
    )


def _gate_deny_result(
    *,
    profile: CollaborativeOperationPolicyProfile,
    collaborative_authority: PolicyDecision,
    reason: str,
    policy_rule_id: str,
    workspace_policy: PolicyDecision | None = None,
    resource_policy: PolicyDecision | None = None,
    runtime_policy: PolicyDecision | None = None,
) -> CollaborativeWorkEnforcementResult:
    applicability = PolicyCompositionApplicability(
        workspace_policy=profile.workspace_policy_applicability,
        resource_policy=profile.resource_policy_applicability,
        runtime_policy=profile.runtime_policy_applicability,
    )
    composed = compose_policy_decisions(
        PolicyCompositionInput(
            collaborative_authority=collaborative_authority,
            workspace_policy=workspace_policy,
            resource_policy=resource_policy,
            runtime_policy=runtime_policy,
            applicability=applicability,
        ),
    )
    audit_payload: dict[str, Any] = {
        "schema": "enforcement_gate_audit.v1",
        "gate_failure": True,
        "operation_id": profile.operation_id,
        "profile_revision": profile.revision,
        "authority_scope": profile.authority_scope,
        "policy_rule_id": policy_rule_id,
    }
    contributing_layers = composed.decision.audit_payload.get("contributing_layers")
    if contributing_layers:
        audit_payload["contributing_layers"] = contributing_layers
    deny = PolicyDecision(
        action=PolicyAction.DENY,
        reason=reason,
        policy_rule_id=policy_rule_id,
        audit_payload=audit_payload,
    )
    composition = PolicyCompositionResult(
        decision=deny,
        collaborative_authority=composed.collaborative_authority,
        workspace_policy=composed.workspace_policy,
        resource_policy=composed.resource_policy,
        runtime_policy=composed.runtime_policy,
        determining_layer=None,
    )
    return CollaborativeWorkEnforcementResult(
        operation_id=profile.operation_id,
        profile_revision=profile.revision,
        authority_scope=profile.authority_scope,
        composition=composition,
    )


class CollaborativeWorkEnforcementGate:
    """Orchestrate trusted classification, authority, policy evaluation, and composition."""

    def __init__(
        self,
        *,
        profile_repository: CollaborativeOperationPolicyProfileRepository,
        authority_resolver: CollaborativeWorkAuthorityResolver,
        policy_evaluator: CollaborativePolicyEvaluator,
        runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator,
    ) -> None:
        self._profile_repository = profile_repository
        self._authority_resolver = authority_resolver
        self._policy_evaluator = policy_evaluator
        self._runtime_policy_evaluator = runtime_policy_evaluator

    def evaluate(self, request: CollaborativeWorkEnforcementRequest) -> CollaborativeWorkEnforcementResult:
        profile = self._profile_repository.get_for_operation(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            operation_id=request.operation_id,
        )
        if profile is None:
            return _classification_result(
                operation_id=request.operation_id,
                reason="operation policy profile missing",
                policy_rule_id=_POLICY_RULE_PROFILE_MISSING,
            )
        if profile.status is not CollaborativeOperationPolicyProfileStatus.ACTIVE:
            return _classification_result(
                operation_id=request.operation_id,
                reason="operation policy profile inactive",
                policy_rule_id=_POLICY_RULE_PROFILE_INACTIVE,
            )

        authority_request = EffectiveAuthorityRequest(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            acting_principal_id=request.acting_principal_id,
            requested_authority_scopes=(profile.authority_scope,),
            delegator_principal_id=request.delegator_principal_id,
            resource_scope=request.resource_scope,
            membership=request.membership,
            delegation=request.delegation,
        )
        authority_decision = self._authority_resolver.resolve(authority_request)

        if profile.resource_requirement is OperationPolicyRequirement.REQUIRED:
            if request.resource_scope is None:
                return _gate_deny_result(
                    profile=profile,
                    collaborative_authority=authority_decision.decision,
                    reason="resource scope required for operation",
                    policy_rule_id=_POLICY_RULE_RESOURCE_REQUIRED_MISSING,
                )

        workspace_policy: PolicyDecision | None = None
        if profile.workspace_policy_applicability is PolicyLayerApplicability.REQUIRED:
            workspace_policy = self._policy_evaluator.evaluate_workspace_policy(
                tenant_id=profile.tenant_id,
                workspace_id=profile.workspace_id,
                authority_scope=profile.authority_scope,
            )

        resource_policy: PolicyDecision | None = None
        if profile.resource_policy_applicability is PolicyLayerApplicability.REQUIRED:
            resource_scope = request.resource_scope
            if resource_scope is None:
                return _gate_deny_result(
                    profile=profile,
                    collaborative_authority=authority_decision.decision,
                    reason="resource scope required for resource policy",
                    policy_rule_id=_POLICY_RULE_RESOURCE_REQUIRED_MISSING,
                    workspace_policy=workspace_policy,
                )
            resource_policy = self._policy_evaluator.evaluate_resource_policy(
                tenant_id=profile.tenant_id,
                workspace_id=profile.workspace_id,
                resource_scope=resource_scope,
                authority_scope=profile.authority_scope,
            )

        runtime_policy: PolicyDecision | None = None
        if profile.runtime_policy_applicability is PolicyLayerApplicability.REQUIRED:
            runtime_request = request.meaningful_side_effect_request
            if runtime_request is None:
                return _gate_deny_result(
                    profile=profile,
                    collaborative_authority=authority_decision.decision,
                    reason="meaningful side-effect request required",
                    policy_rule_id=_POLICY_RULE_RUNTIME_REQUEST_MISSING,
                    workspace_policy=workspace_policy,
                    resource_policy=resource_policy,
                )
            mismatch = self._validate_runtime_request(
                profile=profile,
                request=runtime_request,
                acting_principal_id=request.acting_principal_id,
                resource_scope=request.resource_scope,
            )
            if mismatch is not None:
                return _gate_deny_result(
                    profile=profile,
                    collaborative_authority=authority_decision.decision,
                    reason=mismatch,
                    policy_rule_id=_POLICY_RULE_RUNTIME_IDENTITY_MISMATCH,
                    workspace_policy=workspace_policy,
                    resource_policy=resource_policy,
                )
            runtime_policy = self._runtime_policy_evaluator.evaluate_meaningful_side_effect(
                runtime_request,
            )

        applicability = PolicyCompositionApplicability(
            workspace_policy=profile.workspace_policy_applicability,
            resource_policy=profile.resource_policy_applicability,
            runtime_policy=profile.runtime_policy_applicability,
        )
        composition = compose_policy_decisions(
            PolicyCompositionInput(
                collaborative_authority=authority_decision.decision,
                workspace_policy=workspace_policy,
                resource_policy=resource_policy,
                runtime_policy=runtime_policy,
                applicability=applicability,
            ),
        )
        return CollaborativeWorkEnforcementResult(
            operation_id=profile.operation_id,
            profile_revision=profile.revision,
            authority_scope=profile.authority_scope,
            composition=composition,
        )

    @staticmethod
    def _validate_runtime_request(
        *,
        profile: CollaborativeOperationPolicyProfile,
        request: MeaningfulSideEffectRequest,
        acting_principal_id: str,
        resource_scope: str | None,
    ) -> str | None:
        if request.action != profile.operation_id:
            return "runtime request action does not match operation profile"
        if request.tenant_id is not None and request.tenant_id != profile.tenant_id:
            return "runtime request tenant_id does not match trusted tenant"
        if profile.resource_requirement is OperationPolicyRequirement.REQUIRED:
            if resource_scope is None:
                return "resource scope required for runtime request validation"
            if request.resource != resource_scope:
                return "runtime request resource does not match trusted resource scope"
        if request.principal_id is not None and request.principal_id != acting_principal_id:
            return "runtime request principal_id does not match acting principal"
        if profile.meaningful_side_effect_requirement is OperationPolicyRequirement.REQUIRED:
            if not (request.principal_id or "").strip():
                return "runtime request principal_id required for meaningful side effect"
        if not (request.task_id or "").strip() or not (request.run_id or "").strip():
            return "runtime request task_id and run_id are required"
        return None
