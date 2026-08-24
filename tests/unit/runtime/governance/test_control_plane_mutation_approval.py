# © Artur Czarnecki. All rights reserved.

"""CPM-APPROVAL-1 — canonical scoped control-plane mutation approval proofs (CPMA-1..30)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationScope,
    ControlPlaneMutationRequest,
    ControlPlaneMutationRisk,
    GovernanceEvaluationPoint,
    authorization_scope_for_request,
    control_plane_mutation_request_digest,
    evidence_from_request_and_decision,
)
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.control_plane_mutation_approval import (
    ApprovalConsumingControlPlaneMutationEvaluator,
    ControlPlaneMutationApprovalCoordinator,
    ControlPlaneMutationApprovalError,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_TASK_ID = TaskId("task_01234567890123456789012345678901")
_RUN_ID = RunId("run_01234567890123456789012345678901")


@dataclass
class _RecordingEvaluator:
    decision: PolicyDecision = field(
        default_factory=lambda: PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        )
    )
    calls: list[ControlPlaneMutationRequest] = field(default_factory=list)

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        self.calls.append(request)
        return self.decision


@dataclass
class _FakeDomainExecutor:
    executions: list[ControlPlaneMutationRequest] = field(default_factory=list)

    def execute(self, request: ControlPlaneMutationRequest) -> None:
        self.executions.append(request)


def _service_principal(
    *,
    tenant_id: str = _TENANT,
    user_id: str = "svc-capacity",
) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.SERVICE,
        auth_subject=user_id,
    )


def _human_approver(
    *,
    tenant_id: str = _TENANT,
    user_id: str = "human-operator",
) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


def _request(
    *,
    mutation_id: str = "mut-abc",
    mutation_type: str = "scale_capacity",
    principal: RequestIdentity | None = None,
    resource_scope: str = "workspace-a",
    resource_type: str = "capacity_target",
    resource_id: str = "pool-a",
    current_revision: str = "5",
    target_revision: str = "6",
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    approval_evidence_ref: str | None = None,
) -> ControlPlaneMutationRequest:
    return ControlPlaneMutationRequest(
        mutation_id=mutation_id,
        mutation_type=mutation_type,
        principal=principal or _service_principal(),
        resource_scope=resource_scope,
        resource_type=resource_type,
        resource_id=resource_id,
        current_revision=current_revision,
        target_revision=target_revision,
        risk_classification=ControlPlaneMutationRisk.HIGH,
        task_id=task_id,
        run_id=run_id,
        approval_evidence_ref=approval_evidence_ref,
    )


def _require_human_evidence(
    request: ControlPlaneMutationRequest,
) -> ControlPlaneMutationAuthorizationEvidence:
    digest = control_plane_mutation_request_digest(request)
    return evidence_from_request_and_decision(
        request,
        decision=PolicyDecision(
            action=PolicyAction.REQUIRE_HUMAN,
            reason="needs human",
            policy_rule_id="rule.hitl",
        ),
        request_digest=digest,
    )


def _scope_for(request: ControlPlaneMutationRequest) -> ControlPlaneMutationAuthorizationScope:
    return authorization_scope_for_request(request)


def _create_grant(
    coordinator: ControlPlaneMutationApprovalCoordinator,
    request: ControlPlaneMutationRequest,
    *,
    approver: RequestIdentity | None = None,
) -> str:
    scope = _scope_for(request)
    evidence = _require_human_evidence(request)
    grant = coordinator.create_approval_grant(
        approver=approver or _human_approver(),
        service_principal=request.principal,
        scope=scope,
        authorization_evidence=evidence,
    )
    return grant.grant_id


def test_cpma_1_valid_user_approver_exact_scope_creates_grant() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    request = _request()
    grant_id = _create_grant(coordinator, request)
    grant = coordinator.get_grant(grant_id)
    assert grant is not None
    assert grant.mutation_id == request.mutation_id
    assert grant.approver_principal_type is PrincipalType.USER


def test_cpma_2_non_user_approver_rejected() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    request = _request()
    with pytest.raises(ControlPlaneMutationApprovalError, match="USER principal"):
        coordinator.create_approval_grant(
            approver=_service_principal(user_id="not-human"),
            service_principal=request.principal,
            scope=_scope_for(request),
            authorization_evidence=_require_human_evidence(request),
        )


def test_cpma_3_missing_approver_identity_rejected() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    request = _request()
    bad_approver = RequestIdentity(
        tenant_id=_TENANT,
        user_id=None,
        principal_type=PrincipalType.USER,
        auth_subject=None,
    )
    with pytest.raises(ControlPlaneMutationApprovalError, match="identity required"):
        coordinator.create_approval_grant(
            approver=bad_approver,
            service_principal=request.principal,
            scope=_scope_for(request),
            authorization_evidence=_require_human_evidence(request),
        )


def test_cpma_4_wrong_tenant_approver_rejected() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    request = _request()
    with pytest.raises(ControlPlaneMutationApprovalError, match="tenant_id mismatch"):
        coordinator.create_approval_grant(
            approver=_human_approver(tenant_id="other-tenant"),
            service_principal=request.principal,
            scope=_scope_for(request),
            authorization_evidence=_require_human_evidence(request),
        )


def test_cpma_5_evidence_mutation_id_mismatch_rejected() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    request = _request()
    evidence = _require_human_evidence(request)
    evidence = evidence.model_copy(update={"mutation_id": "mut-other"})
    with pytest.raises(ControlPlaneMutationApprovalError, match="scope mismatch"):
        coordinator.create_approval_grant(
            approver=_human_approver(),
            service_principal=request.principal,
            scope=_scope_for(request),
            authorization_evidence=evidence,
        )


def test_cpma_6_evidence_scope_resource_mismatch_rejected() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    request = _request()
    evidence = _require_human_evidence(request)
    evidence = evidence.model_copy(update={"resource_id": "pool-b"})
    with pytest.raises(ControlPlaneMutationApprovalError, match="scope mismatch"):
        coordinator.create_approval_grant(
            approver=_human_approver(),
            service_principal=request.principal,
            scope=_scope_for(request),
            authorization_evidence=evidence,
        )


def test_cpma_7_evidence_scope_current_revision_mismatch_rejected() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    request = _request()
    evidence = _require_human_evidence(request)
    evidence = evidence.model_copy(update={"current_revision": "99"})
    with pytest.raises(ControlPlaneMutationApprovalError, match="scope mismatch"):
        coordinator.create_approval_grant(
            approver=_human_approver(),
            service_principal=request.principal,
            scope=_scope_for(request),
            authorization_evidence=evidence,
        )


def test_cpma_8_evidence_scope_target_revision_mismatch_rejected() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    request = _request()
    evidence = _require_human_evidence(request)
    evidence = evidence.model_copy(update={"target_revision": "99"})
    with pytest.raises(ControlPlaneMutationApprovalError, match="scope mismatch"):
        coordinator.create_approval_grant(
            approver=_human_approver(),
            service_principal=request.principal,
            scope=_scope_for(request),
            authorization_evidence=evidence,
        )


def _approval_evaluator(
    coordinator: ControlPlaneMutationApprovalCoordinator,
) -> ApprovalConsumingControlPlaneMutationEvaluator:
    return ApprovalConsumingControlPlaneMutationEvaluator(
        inner=_RecordingEvaluator(),
        coordinator=coordinator,
    )


def test_cpma_9_grant_exact_request_match_consumed_and_allow() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request()
    grant_id = _create_grant(coordinator, original)
    resumed = original.model_copy(update={"approval_evidence_ref": grant_id})
    decision = evaluator.evaluate(resumed)
    assert decision.action is PolicyAction.ALLOW
    assert decision.reason == "scoped_human_approval_consumed"
    assert coordinator.is_consumed(grant_id)


def test_cpma_10_second_consume_denies() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request()
    grant_id = _create_grant(coordinator, original)
    resumed = original.model_copy(update={"approval_evidence_ref": grant_id})
    assert evaluator.evaluate(resumed).action is PolicyAction.ALLOW
    second = evaluator.evaluate(resumed)
    assert second.action is PolicyAction.DENY
    assert second.reason == "approval_evidence_invalid_or_consumed"


def test_cpma_11_wrong_mutation_id_does_not_consume() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request()
    grant_id = _create_grant(coordinator, original)
    bad = original.model_copy(
        update={"mutation_id": "mut-other", "approval_evidence_ref": grant_id}
    )
    assert evaluator.evaluate(bad).action is PolicyAction.DENY
    assert not coordinator.is_consumed(grant_id)


def test_cpma_12_wrong_resource_does_not_consume() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request()
    grant_id = _create_grant(coordinator, original)
    bad = original.model_copy(
        update={"resource_id": "pool-b", "approval_evidence_ref": grant_id}
    )
    assert evaluator.evaluate(bad).action is PolicyAction.DENY
    assert not coordinator.is_consumed(grant_id)


def test_cpma_13_wrong_current_revision_does_not_consume() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request()
    grant_id = _create_grant(coordinator, original)
    bad = original.model_copy(
        update={"current_revision": "99", "approval_evidence_ref": grant_id}
    )
    assert evaluator.evaluate(bad).action is PolicyAction.DENY
    assert not coordinator.is_consumed(grant_id)


def test_cpma_14_wrong_target_revision_does_not_consume() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request()
    grant_id = _create_grant(coordinator, original)
    bad = original.model_copy(
        update={"target_revision": "99", "approval_evidence_ref": grant_id}
    )
    assert evaluator.evaluate(bad).action is PolicyAction.DENY
    assert not coordinator.is_consumed(grant_id)


def test_cpma_15_wrong_service_principal_does_not_consume() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request()
    grant_id = _create_grant(coordinator, original)
    bad = original.model_copy(
        update={
            "principal": _service_principal(user_id="other-svc"),
            "approval_evidence_ref": grant_id,
        }
    )
    assert evaluator.evaluate(bad).action is PolicyAction.DENY
    assert not coordinator.is_consumed(grant_id)


def test_cpma_16_wrong_approval_evidence_ref_does_not_consume() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request()
    grant_id = _create_grant(coordinator, original)
    bad = original.model_copy(update={"approval_evidence_ref": "cpm-grant:wrong"})
    assert evaluator.evaluate(bad).action is PolicyAction.DENY
    assert not coordinator.is_consumed(grant_id)


def test_cpma_17_task_id_exact_match_allows() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request(task_id=_TASK_ID, run_id=_RUN_ID)
    grant_id = _create_grant(coordinator, original)
    resumed = original.model_copy(update={"approval_evidence_ref": grant_id})
    assert evaluator.evaluate(resumed).action is PolicyAction.ALLOW


def test_cpma_18_task_id_mismatch_does_not_consume() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request(task_id=_TASK_ID, run_id=_RUN_ID)
    grant_id = _create_grant(coordinator, original)
    other_task = TaskId("task_98765432109876543210987654321098")
    bad = original.model_copy(
        update={"task_id": other_task, "approval_evidence_ref": grant_id}
    )
    assert evaluator.evaluate(bad).action is PolicyAction.DENY
    assert not coordinator.is_consumed(grant_id)


def test_cpma_19_run_id_exact_match_allows() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request(task_id=_TASK_ID, run_id=_RUN_ID)
    grant_id = _create_grant(coordinator, original)
    resumed = original.model_copy(update={"approval_evidence_ref": grant_id})
    assert evaluator.evaluate(resumed).action is PolicyAction.ALLOW


def test_cpma_20_run_id_mismatch_does_not_consume() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request(task_id=_TASK_ID, run_id=_RUN_ID)
    grant_id = _create_grant(coordinator, original)
    other_run = RunId("run_98765432109876543210987654321098")
    bad = original.model_copy(
        update={"run_id": other_run, "approval_evidence_ref": grant_id}
    )
    assert evaluator.evaluate(bad).action is PolicyAction.DENY
    assert not coordinator.is_consumed(grant_id)


def test_cpma_21_none_task_run_exact_matching_works() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request(task_id=None, run_id=None)
    grant_id = _create_grant(coordinator, original)
    resumed = original.model_copy(update={"approval_evidence_ref": grant_id})
    assert evaluator.evaluate(resumed).action is PolicyAction.ALLOW


def test_cpma_22_resumed_request_same_logical_digest_as_original() -> None:
    original = _request()
    grant_id = "cpm-grant:example"
    resumed = original.model_copy(update={"approval_evidence_ref": grant_id})
    assert control_plane_mutation_request_digest(original) == control_plane_mutation_request_digest(
        resumed
    )


def test_cpma_23_changing_material_field_changes_digest() -> None:
    base = _request()
    changed = base.model_copy(update={"mutation_id": "mut-changed"})
    assert control_plane_mutation_request_digest(base) != control_plane_mutation_request_digest(
        changed
    )


def test_cpma_24_mismatched_attempt_does_not_consume_valid_grant() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request()
    grant_id = _create_grant(coordinator, original)
    bad = original.model_copy(
        update={"resource_id": "pool-b", "approval_evidence_ref": grant_id}
    )
    assert evaluator.evaluate(bad).action is PolicyAction.DENY
    good = original.model_copy(update={"approval_evidence_ref": grant_id})
    assert evaluator.evaluate(good).action is PolicyAction.ALLOW


def test_cpma_25_human_denial_exact_scope_retained() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    request = _request()
    scope = _scope_for(request)
    evidence = _require_human_evidence(request)
    denial = coordinator.record_denial(
        approver=_human_approver(),
        scope=scope,
        authorization_evidence=evidence,
    )
    assert denial.mutation_id == scope.mutation_id
    assert denial.resource_id == scope.resource_id
    assert denial.request_digest == evidence.request_digest
    assert coordinator.get_denial(request.mutation_id) is denial


def test_cpma_26_denial_never_acts_as_approval() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    request = _request()
    coordinator.record_denial(
        approver=_human_approver(),
        scope=_scope_for(request),
        authorization_evidence=_require_human_evidence(request),
    )
    fake_ref = "cpm-grant:not-a-grant"
    resumed = request.model_copy(update={"approval_evidence_ref": fake_ref})
    assert evaluator.evaluate(resumed).action is PolicyAction.DENY


def test_cpma_27_unknown_grant_denies() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    request = _request(approval_evidence_ref="cpm-grant:missing")
    assert evaluator.evaluate(request).action is PolicyAction.DENY


def test_cpma_28_approval_consumption_returns_authorization_only_domain_execution_remains_explicit() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    domain_executor = _FakeDomainExecutor()
    original = _request()
    grant_id = _create_grant(coordinator, original)
    resumed = original.model_copy(update={"approval_evidence_ref": grant_id})

    decision = evaluator.evaluate(resumed)
    assert decision.action is PolicyAction.ALLOW
    assert decision.reason == "scoped_human_approval_consumed"
    assert domain_executor.executions == []

    domain_executor.execute(resumed)
    assert len(domain_executor.executions) == 1
    assert domain_executor.executions[0].mutation_id == original.mutation_id


def test_cpma_29_generic_non_ecp_control_plane_mutation_can_use_scoped_approval_primitive_end_to_end() -> None:
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request(
        mutation_id="mut-credential-rotate",
        mutation_type="rotate_service_credentials",
        resource_type="service_credential",
        resource_id="credential-set-a",
        resource_scope="tenant-a/security",
        current_revision="revision-1",
        target_revision="revision-2",
        principal=_service_principal(user_id="svc-security"),
    )
    grant_id = _create_grant(coordinator, original)
    grant = coordinator.get_grant(grant_id)
    assert grant is not None
    assert grant.mutation_id == original.mutation_id
    assert grant.resource_type == "service_credential"
    assert grant.resource_id == "credential-set-a"

    resumed = original.model_copy(update={"approval_evidence_ref": grant_id})
    assert resumed.mutation_id == original.mutation_id
    assert resumed.resource_type == "service_credential"
    assert resumed.resource_id == "credential-set-a"

    decision = evaluator.evaluate(resumed)
    assert decision.action is PolicyAction.ALLOW
    assert coordinator.is_consumed(grant_id)

    second = evaluator.evaluate(resumed)
    assert second.action is PolicyAction.DENY


def test_cpma_30_consumption_is_one_attempt_not_success_reuse() -> None:
    """Grant consumed at evaluation — provider failure cannot reuse approval."""
    coordinator = ControlPlaneMutationApprovalCoordinator()
    evaluator = _approval_evaluator(coordinator)
    original = _request()
    grant_id = _create_grant(coordinator, original)
    resumed = original.model_copy(update={"approval_evidence_ref": grant_id})
    first = evaluator.evaluate(resumed)
    assert first.action is PolicyAction.ALLOW
    assert coordinator.is_consumed(grant_id)
    # Simulated provider/stale failure path — grant is gone; retry needs fresh approval.
    retry = evaluator.evaluate(resumed)
    assert retry.action is PolicyAction.DENY
    assert coordinator.get_grant(grant_id) is None
