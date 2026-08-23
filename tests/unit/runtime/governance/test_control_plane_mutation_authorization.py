# © Artur Czarnecki. All rights reserved.

"""CLA-CONTROL-PLANE-GOVERNANCE-INTEGRITY-FOUNDATION boundary tests (CP1–CP16)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationRequest,
    ControlPlaneMutationRisk,
    GovernanceEvaluationPoint,
    control_plane_mutation_request_digest,
)
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

pytestmark = pytest.mark.unit

_TASK_ID = TaskId("task_01234567890123456789012345678901")
_RUN_ID = RunId("run_01234567890123456789012345678901")


@dataclass
class _FakeEvaluator:
    decision: PolicyDecision = field(
        default_factory=lambda: PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
    )
    calls: list[ControlPlaneMutationRequest] = field(default_factory=list)
    raise_error: bool = False

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        if self.raise_error:
            raise RuntimeError("evaluator exploded")
        return self.decision


def _principal(*, tenant_id: str = "tenant-a", user_id: str = "user-1") -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


def _request(
    *,
    mutation_id: str = "mut-abc",
    current_revision: str = "5",
    target_revision: str = "6",
    principal: RequestIdentity | None = None,
    resource_scope: str = "workspace-a",
    resource_type: str = "ahi_configuration",
    resource_id: str = "adaptive-profile-1",
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    risk: ControlPlaneMutationRisk = ControlPlaneMutationRisk.MEDIUM,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type="apply_configuration",
        principal=principal or _principal(),
        resource_scope=resource_scope,
        resource_type=resource_type,
        resource_id=resource_id,
        current_revision=current_revision,
        target_revision=target_revision,
        risk_classification=risk,
        task_id=task_id,
        run_id=run_id,
    )


def test_cp1_valid_request_allow() -> None:
    evaluator = _FakeEvaluator(
        decision=PolicyDecision(action=PolicyAction.ALLOW, reason="allowed")
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    result = boundary.authorize(_request())
    assert result.permitted is True
    assert result.decision.action is PolicyAction.ALLOW
    assert len(evaluator.calls) == 1


def test_cp2_deny_preserved() -> None:
    evaluator = _FakeEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="policy_deny",
            policy_rule_id="rule.deny",
        )
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    result = boundary.authorize(_request())
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY
    assert result.decision.reason == "policy_deny"


def test_cp3_require_human_carries_scope_identity() -> None:
    evaluator = _FakeEvaluator(
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs_approval",
            policy_rule_id="rule.hitl",
        )
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    request = _request(
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        mutation_id="mut-hitl",
    )
    result = boundary.authorize(request)
    assert result.permitted is False
    assert result.requires_governed_continuation is True
    scope = result.authorization_scope
    assert scope is not None
    assert scope.mutation_id == "mut-hitl"
    assert scope.resource_type == "ahi_configuration"
    assert scope.resource_id == "adaptive-profile-1"
    assert scope.current_revision == "5"
    assert scope.target_revision == "6"
    assert scope.task_id == str(_TASK_ID)
    assert scope.run_id == str(_RUN_ID)


def test_cp4_missing_principal_fail_closed() -> None:
    evaluator = _FakeEvaluator()
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    request = ControlPlaneMutationRequest.model_construct(
        mutation_id="mut-1",
        mutation_type="apply",
        principal=None,
        resource_scope="workspace-a",
        resource_type="task_control",
        resource_id="task-123",
        current_revision="1",
        target_revision="2",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )
    result = boundary.authorize(request)
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY
    assert result.validation_failed is True
    assert evaluator.calls == []


def test_cp5_missing_tenant_fail_closed() -> None:
    evaluator = _FakeEvaluator()
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    principal = RequestIdentity.model_construct(
        tenant_id="",
        user_id="user-1",
        principal_type=PrincipalType.USER,
        auth_subject="user-1",
    )
    request = _request(principal=principal)
    result = boundary.authorize(request)
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY
    assert result.validation_failed is True
    assert evaluator.calls == []


def test_cp5b_missing_resource_scope_fail_closed() -> None:
    evaluator = _FakeEvaluator()
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    request = ControlPlaneMutationRequest.model_construct(
        mutation_id="mut-1",
        mutation_type="apply",
        principal=_principal(),
        resource_scope="",
        resource_type="capacity_target",
        resource_id="pool-a",
        current_revision="1",
        target_revision="2",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )
    result = boundary.authorize(request)
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY
    assert evaluator.calls == []


def test_cp6_missing_resource_fail_closed() -> None:
    evaluator = _FakeEvaluator()
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    request = ControlPlaneMutationRequest.model_construct(
        mutation_id="mut-1",
        mutation_type="apply",
        principal=_principal(),
        resource_scope="workspace-a",
        resource_type="",
        resource_id="",
        current_revision="1",
        target_revision="2",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )
    result = boundary.authorize(request)
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY
    assert evaluator.calls == []


def test_cp7_missing_mutation_identity_fail_closed() -> None:
    evaluator = _FakeEvaluator()
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    request = ControlPlaneMutationRequest.model_construct(
        mutation_id="",
        mutation_type="apply",
        principal=_principal(),
        resource_scope="workspace-a",
        resource_type="plugin_admission",
        resource_id="plugin-x",
        current_revision="1",
        target_revision="2",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )
    result = boundary.authorize(request)
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY
    assert evaluator.calls == []


def test_cp8_revision_binding_produces_distinct_request_digest() -> None:
    first = _request(current_revision="5", target_revision="6")
    second = _request(current_revision="6", target_revision="7")
    assert control_plane_mutation_request_digest(first) != control_plane_mutation_request_digest(
        second
    )


def test_cp9_risk_survives_boundary_roundtrip() -> None:
    evaluator = _FakeEvaluator()
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    result = boundary.authorize(_request(risk=ControlPlaneMutationRisk.CRITICAL))
    assert result.evidence.risk_classification is ControlPlaneMutationRisk.CRITICAL


def test_cp10_evaluator_failure_fail_closed() -> None:
    evaluator = _FakeEvaluator(raise_error=True)
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    result = boundary.authorize(_request())
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY
    assert "evaluator_failure" in result.decision.reason


def test_cp11_modify_fail_closed_to_deny() -> None:
    evaluator = _FakeEvaluator(
        decision=PolicyDecision(action=PolicyAction.MODIFY, reason="modify")
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    result = boundary.authorize(_request())
    assert result.permitted is False
    assert result.decision.action is PolicyAction.DENY


def test_cp12_authorize_does_not_execute_domain_mutation() -> None:
    mutation_counter = {"count": 0}

    def _domain_mutation_executor() -> None:
        mutation_counter["count"] += 1

    evaluator = _FakeEvaluator(
        decision=PolicyDecision(action=PolicyAction.ALLOW, reason="allowed")
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    boundary.authorize(_request())
    assert mutation_counter["count"] == 0
    _domain_mutation_executor()
    assert mutation_counter["count"] == 1


def test_cp14_identity_distinction_task_run_mutation_resource() -> None:
    request = _request(
        mutation_id="mut-distinct",
        resource_id="worker-pool-a",
        task_id=_TASK_ID,
        run_id=_RUN_ID,
    )
    assert request.mutation_id != str(request.task_id)
    assert str(request.task_id) != str(request.run_id)
    assert request.resource_id != request.mutation_id
    assert request.resource_id != str(request.task_id)


def test_cp15_execution_identity_from_canonical_types() -> None:
    request = _request(task_id=_TASK_ID, run_id=_RUN_ID)
    assert request.task_id == _TASK_ID
    assert request.run_id == _RUN_ID
    assert str(request.run_id).startswith("run_")
    assert str(request.task_id).startswith("task_")


def test_cp15_rejects_task_id_without_run_id() -> None:
    with pytest.raises(ValueError, match="task_id and run_id"):
        _request(task_id=_TASK_ID, run_id=None)


def test_cp16_evaluation_point_is_control_plane_mutation() -> None:
    evaluator = _FakeEvaluator()
    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=evaluator)
    result = boundary.authorize(_request())
    assert (
        result.evidence.evaluation_point
        is GovernanceEvaluationPoint.CONTROL_PLANE_MUTATION
    )
