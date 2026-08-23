# © Artur Czarnecki. All rights reserved.

"""Canonical pre-mutation authorization boundary for control-plane operations (CLA-04).

Evaluation only — domain executors perform mutations after an ALLOW decision.
"""

from __future__ import annotations

from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationResult,
    ControlPlaneMutationPolicyEvaluator,
    ControlPlaneMutationRequest,
    authorization_scope_for_request,
    control_plane_mutation_request_digest,
    evidence_from_request_and_decision,
)
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision


class ControlPlaneMutationAuthorizationBoundary:
    """Shared authorization boundary for CONTROL_PLANE_MUTATION evaluation points."""

    def __init__(self, *, evaluator: ControlPlaneMutationPolicyEvaluator) -> None:
        self._evaluator = evaluator

    @property
    def evaluator(self) -> ControlPlaneMutationPolicyEvaluator:
        return self._evaluator

    def authorize(
        self,
        request: ControlPlaneMutationRequest,
    ) -> ControlPlaneMutationAuthorizationResult:
        validation_reason = self._validate_request(request)
        if validation_reason is not None:
            return self._fail_closed(
                request,
                reason=validation_reason,
                validation_failed=True,
            )

        try:
            decision = self._evaluator.evaluate(request)
        except Exception:
            return self._fail_closed(
                request,
                reason="evaluator_failure",
                validation_failed=False,
            )

        if decision.action is PolicyAction.MODIFY:
            return self._fail_closed(
                request,
                reason="modify_not_supported_for_control_plane_mutation",
                decision=decision,
                validation_failed=False,
            )

        if decision.action not in PolicyAction:
            return self._fail_closed(
                request,
                reason="unsupported_policy_action",
                validation_failed=False,
            )

        request_digest = control_plane_mutation_request_digest(request)
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
        authorization_scope = (
            authorization_scope_for_request(request)
            if requires_continuation
            else None
        )
        return ControlPlaneMutationAuthorizationResult(
            permitted=permitted,
            decision=decision,
            evidence=evidence,
            requires_governed_continuation=requires_continuation,
            authorization_scope=authorization_scope,
            validation_failed=False,
        )

    @staticmethod
    def _validate_request(request: ControlPlaneMutationRequest) -> str | None:
        principal = request.principal
        if principal is None:
            return "missing_principal"
        if not principal.tenant_id.strip():
            return "missing_tenant"
        if not (principal.user_id or principal.auth_subject):
            return "missing_principal_identity"
        if not request.resource_scope.strip():
            return "missing_resource_scope"
        if not request.resource_type.strip() or not request.resource_id.strip():
            return "missing_resource_identity"
        if not request.mutation_id.strip():
            return "missing_mutation_identity"
        if not request.current_revision.strip() or not request.target_revision.strip():
            return "missing_revision_context"
        return None

    def _fail_closed(
        self,
        request: ControlPlaneMutationRequest,
        *,
        reason: str,
        decision: PolicyDecision | None = None,
        validation_failed: bool,
    ) -> ControlPlaneMutationAuthorizationResult:
        deny = decision or PolicyDecision(
            action=PolicyAction.DENY,
            reason=reason,
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="control_plane_mutation.boundary",
            decision_id=f"cpm-deny:{reason}",
        )
        if deny.action is not PolicyAction.DENY:
            deny = PolicyDecision(
                action=PolicyAction.DENY,
                reason=reason,
                enforcement_level=EnforcementLevel.MANDATORY,
                policy_rule_id=deny.policy_rule_id or "control_plane_mutation.boundary",
                decision_id=deny.decision_id or f"cpm-deny:{reason}",
                policy_bundle_id=deny.policy_bundle_id,
                policy_bundle_version=deny.policy_bundle_version,
                policy_bundle_digest=deny.policy_bundle_digest,
            )

        request_digest = self._safe_request_digest(request)
        evidence = self._safe_evidence(request, decision=deny, request_digest=request_digest)
        return ControlPlaneMutationAuthorizationResult(
            permitted=False,
            decision=deny,
            evidence=evidence,
            requires_governed_continuation=False,
            authorization_scope=None,
            validation_failed=validation_failed,
        )

    @staticmethod
    def _safe_request_digest(request: ControlPlaneMutationRequest) -> str:
        try:
            return control_plane_mutation_request_digest(request)
        except Exception:
            return "sha256:0000000000000000000000000000000000000000000000000000000000000000"

    @staticmethod
    def _safe_evidence(
        request: ControlPlaneMutationRequest,
        *,
        decision: PolicyDecision,
        request_digest: str,
    ) -> ControlPlaneMutationAuthorizationEvidence:
        try:
            return evidence_from_request_and_decision(
                request,
                decision=decision,
                request_digest=request_digest,
            )
        except Exception:
            tenant_id = "unknown"
            if request.principal is not None and request.principal.tenant_id.strip():
                tenant_id = request.principal.tenant_id
            return ControlPlaneMutationAuthorizationEvidence(
                request_digest=request_digest,
                mutation_id=getattr(request, "mutation_id", "") or "unknown",
                mutation_type=getattr(request, "mutation_type", "") or "unknown",
                tenant_id=tenant_id,
                resource_scope=getattr(request, "resource_scope", "") or "unknown",
                resource_type=getattr(request, "resource_type", "") or "unknown",
                resource_id=getattr(request, "resource_id", "") or "unknown",
                current_revision=getattr(request, "current_revision", "") or "unknown",
                target_revision=getattr(request, "target_revision", "") or "unknown",
                risk_classification=request.risk_classification,
                principal_type=(
                    request.principal.principal_type
                    if request.principal is not None
                    else PrincipalType.USER
                ),
                policy_action=decision.action,
                policy_rule_id=decision.policy_rule_id,
                policy_decision_id=decision.decision_id,
            )
