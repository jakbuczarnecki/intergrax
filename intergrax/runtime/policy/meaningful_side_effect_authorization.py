# © Artur Czarnecki. All rights reserved.

"""Canonical pre-side-effect authorization boundary (COLLAB-WORK-1H).

Invokes ``CollaborativeWorkEnforcementGate`` immediately before a proposed
meaningful side effect may proceed. Evaluation only — execution remains owned
by the caller/runtime layer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.contracts.canonical_inner_governance import (
    CanonicalInnerExecutionGuardPort,
    CanonicalInnerGovernanceViolation,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    PolicyCompositionResult,
)
from intergrax.contracts.decision_requirement_policy import (
    DecisionRequirement,
    DecisionRequirementContext,
    DecisionRequirementPolicy,
)
from intergrax.contracts.evaluated_policy_decision import request_digest_for_payload
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernedExecutionEvaluationPoint,
    build_governance_fact_from_policy_decision,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
    classify_decision_requirement,
)
from intergrax.runtime.human.governed_continuation_bridge import (
    apply_governed_continuation_pause,
    compose_governed_continuation_from_enforcement,
)
from intergrax.runtime.human.governed_continuation_grant import (
    GovernedContinuationGrantCoordinator,
    matches_current_requirement,
)
from intergrax.runtime.decision_governance_material import (
    assert_decision_governance_material_bound,
)
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_lifecycle import TaskLifecycle, TaskState

T = TypeVar("T")


class MeaningfulSideEffectAuthorizationBoundary:
    """Shared production boundary for collaborative enforcement before side effects."""

    def __init__(
        self,
        *,
        enforcement_gate: CollaborativeWorkEnforcementGate,
        inner_execution_guard: CanonicalInnerExecutionGuardPort,
        decision_requirement_policy: DecisionRequirementPolicy | None = None,
        governance_evidence_recorder: GovernanceEvidenceRecorder | None = None,
    ) -> None:
        self._enforcement_gate = enforcement_gate
        self._inner_execution_guard = inner_execution_guard
        self._decision_requirement_policy = (
            decision_requirement_policy
            if decision_requirement_policy is not None
            else PermissiveDecisionRequirementPolicy()
        )
        self._governance_evidence_recorder = governance_evidence_recorder

    def _record_governance_evidence(
        self,
        request: CollaborativeWorkEnforcementRequest,
        decision: PolicyDecision,
    ) -> None:
        recorder = self._governance_evidence_recorder
        if recorder is None or recorder.persistence is None:
            return
        if decision.action not in (
            PolicyAction.ALLOW,
            PolicyAction.DENY,
            PolicyAction.REQUIRE_HUMAN,
        ):
            return
        side_effect = request.meaningful_side_effect_request
        tenant_id = request.tenant_id
        workspace_id = request.workspace_id
        principal_id = request.acting_principal_id
        task_id = None
        run_id = None
        attempt_id = None
        execution_id = None
        action = request.operation_id
        resource_type = ""
        resource_scope = request.resource_scope or ""
        decision_material = None
        if type(side_effect) is MeaningfulSideEffectRequest:
            if side_effect.tenant_id:
                tenant_id = side_effect.tenant_id
            principal_id = side_effect.principal_id or principal_id
            task_id = side_effect.task_id
            run_id = side_effect.run_id
            attempt_id = side_effect.attempt_id
            execution_id = side_effect.execution_id
            action = side_effect.action
            resource_type = side_effect.kinds[0].value if side_effect.kinds else ""
            resource_scope = side_effect.side_effect_scope_id
            decision_material = side_effect.decision_governance_material
        digest = request_digest_for_payload(
            {
                "operation_id": request.operation_id,
                "resource_scope": request.resource_scope,
                "decision_action": decision.action.value,
                "policy_rule_id": decision.policy_rule_id,
            }
        )
        idempotency_key = f"mse:{digest}:{decision.action.value}"
        fact = build_governance_fact_from_policy_decision(
            evaluation_point=GovernedExecutionEvaluationPoint.MEANINGFUL_SIDE_EFFECT,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            principal_id=principal_id,
            decision=decision,
            request_digest=digest,
            idempotency_key=idempotency_key,
            action=action,
            resource_type=resource_type,
            resource_scope=resource_scope,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            decision_material_ref=decision_material,
        )
        recorder.record(fact)

    def _inner_enforcement_denied(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        reason: str,
    ) -> MeaningfulSideEffectAuthorizationResult:
        deny = PolicyDecision(
            action=PolicyAction.DENY,
            reason=reason,
            policy_rule_id="platform.canonical_inner_enforcement",
        )
        composition = PolicyCompositionResult(
            decision=deny,
            collaborative_authority=deny,
        )
        enforcement_result = CollaborativeWorkEnforcementResult(
            operation_id=request.operation_id,
            authority_scope=request.resource_scope,
            composition=composition,
        )
        result = MeaningfulSideEffectAuthorizationResult(
            permitted=False,
            decision=deny,
            enforcement_result=enforcement_result,
            requires_governed_continuation=False,
            governed_continuation_request=None,
        )
        self._record_governance_evidence(request, deny)
        return result

    def _assert_inner_execution(
        self,
        request: CollaborativeWorkEnforcementRequest,
    ) -> MeaningfulSideEffectAuthorizationResult | None:
        side_effect = request.meaningful_side_effect_request
        if side_effect is None:
            return self._inner_enforcement_denied(
                request,
                reason="meaningful side effect request required for inner enforcement",
            )
        try:
            self._inner_execution_guard.assert_meaningful_side_effect_bound(side_effect)
            requirement_block = self._enforce_decision_requirement(request, side_effect)
            if requirement_block is not None:
                return requirement_block
            assert_decision_governance_material_bound(side_effect)
        except CanonicalInnerGovernanceViolation as exc:
            return self._inner_enforcement_denied(request, reason=exc.reason)
        except RuntimeError as exc:
            return self._inner_enforcement_denied(request, reason=str(exc))
        return None

    def _enforce_decision_requirement(
        self,
        request: CollaborativeWorkEnforcementRequest,
        side_effect: object,
    ) -> MeaningfulSideEffectAuthorizationResult | None:
        try:
            context = DecisionRequirementContext.from_meaningful_side_effect(
                operation_id=request.operation_id,
                resource_scope=request.resource_scope,
                side_effect=side_effect,
            )
            requirement = classify_decision_requirement(
                self._decision_requirement_policy,
                context,
            )
        except Exception as exc:  # noqa: BLE001 — requirement policy must fail closed
            return self._inner_enforcement_denied(
                request,
                reason=f"decision requirement policy evaluation failed: {exc}",
            )
        if requirement in (
            DecisionRequirement.REQUIRED,
            DecisionRequirement.UNDETERMINED,
        ):
            from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest

            if (
                type(side_effect) is not MeaningfulSideEffectRequest
                or side_effect.decision_governance_material is None
            ):
                return self._inner_enforcement_denied(
                    request,
                    reason="decision provenance required but absent",
                )
        return None

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.meaningful_side_effect",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        inner_block = self._assert_inner_execution(request)
        if inner_block is not None:
            return inner_block
        enforcement_result = self._enforcement_gate.evaluate(request)
        decision = enforcement_result.composition.decision
        action = decision.action
        permitted = action is PolicyAction.ALLOW
        requires_continuation = action in (PolicyAction.REQUIRE_HUMAN, PolicyAction.ESCALATE)
        governed_continuation_request = compose_governed_continuation_from_enforcement(
            request,
            decision=decision,
            enforcement_operation_id=enforcement_result.operation_id,
            enforcement_authority_scope=enforcement_result.authority_scope,
            requires_governed_continuation=requires_continuation,
            source_agent_id=source_agent_id,
            source_step_id=source_step_id,
        )
        result = MeaningfulSideEffectAuthorizationResult(
            permitted=permitted,
            decision=decision,
            enforcement_result=enforcement_result,
            requires_governed_continuation=requires_continuation,
            governed_continuation_request=governed_continuation_request,
        )
        self._record_governance_evidence(request, decision)
        return result

    def authorize_and_execute(
        self,
        request: CollaborativeWorkEnforcementRequest,
        execute: Callable[[], T],
        *,
        task: Task | None = None,
        lifecycle: TaskLifecycle | None = None,
        source_agent_id: str = "platform.meaningful_side_effect",
        source_step_id: str | None = None,
        on_authorization: Callable[[MeaningfulSideEffectAuthorizationResult], None] | None = None,
        on_execution_authorized: Callable[[], None] | None = None,
    ) -> T | MeaningfulSideEffectAuthorizationResult:
        """Fresh enforcement evaluation before ``execute``.

        ``PolicyAction.ALLOW`` executes without a grant. Fresh ``DENY`` is absolute —
        approval grants cannot override it. Fresh ``REQUIRE_HUMAN`` executes only when a
        stored grant exactly matches the current requirement and side-effect identity;
        the grant is consumed before ``execute`` (at-most-once approval authorization,
        not exactly-once external execution). Without a matching grant, canonical HITL
        pause is entered when ``lifecycle`` is supplied.
        """
        authorization = self.authorize(
            request,
            source_agent_id=source_agent_id,
            source_step_id=source_step_id,
        )
        if on_authorization is not None:
            on_authorization(authorization)
        action = authorization.decision.action
        enforcement = authorization.enforcement_result
        side_effect = request.meaningful_side_effect_request
        resource_scope = request.resource_scope or enforcement.authority_scope
        operation_id = enforcement.operation_id

        if action is PolicyAction.DENY:
            if task is not None and side_effect is not None:
                GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                    task,
                    side_effect=side_effect,
                    operation_id=operation_id,
                    resource_scope=resource_scope,
                )
            return authorization

        if action is PolicyAction.ALLOW:
            if task is not None and side_effect is not None:
                GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                    task,
                    side_effect=side_effect,
                    operation_id=operation_id,
                    resource_scope=resource_scope,
                )
            if on_execution_authorized is not None:
                on_execution_authorized()
            return execute()

        if action is PolicyAction.REQUIRE_HUMAN:
            if task is None or side_effect is None:
                return authorization

            stored_grant = task.runtime.governance.governed_continuation_grant
            if stored_grant is not None:
                if matches_current_requirement(
                    stored_grant,
                    current_side_effect=side_effect,
                    current_operation_id=operation_id,
                    current_resource_scope=resource_scope,
                    current_decision=authorization.decision,
                ):
                    consumed = GovernedContinuationGrantCoordinator.consume_matching_grant(
                        task,
                        expected_grant_id=stored_grant.grant_id,
                    )
                    if consumed is not None:
                        if on_execution_authorized is not None:
                            on_execution_authorized()
                        return execute()
                GovernedContinuationGrantCoordinator.clear_obsolete_grant_for_proposal(
                    task,
                    side_effect=side_effect,
                    operation_id=operation_id,
                    resource_scope=resource_scope,
                )

            if (
                lifecycle is not None
                and authorization.governed_continuation_request is not None
            ):
                apply_governed_continuation_pause(
                    task,
                    authorization.governed_continuation_request,
                )
                lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)
            return authorization

        return authorization
