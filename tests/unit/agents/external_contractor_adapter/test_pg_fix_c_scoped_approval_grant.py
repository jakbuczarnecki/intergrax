# © Artur Czarnecki. All rights reserved.

"""PG-FIX-C — exact scoped approval grant consumption on External Work path."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from external_contractor_adapter.external_work_adapter import (
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_SCOPE_DESCRIPTION,
    META_SCOPE_DIGEST,
    META_WORKSPACE_REF,
    ExternalWorkAdapter,
)
from external_contractor_adapter.side_effect_actions import ACTION_CREATE_EXTERNAL_WORK
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from external_contractor_adapter.tests.fakes.external_work_authorization_boundary import (
    seed_external_work_authorization_boundary,
)
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    CreateCollaborativeOperationPolicyProfileCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfileStatus,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
    WorkspaceMembershipRole,
)
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.human.governed_continuation_bridge import (
    bridge_governed_continuation_to_execution_result,
)
from intergrax.runtime.human.governed_continuation_grant import (
    GovernedContinuationGrantCoordinator,
    GovernedContinuationGrantError,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.contracts.governed_continuation import ContinuationReason, GovernedContinuationRequest
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from intergrax.runtime.task.task import Task, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_TASK_ID = mint_task_id()
_RUN_ID = "run-pg-fix-c"
_OTHER_TASK = mint_task_id()
_OTHER_RUN = "run-pg-fix-c-other"
_PRINCIPAL = "principal-pg-c"
_DIGEST = "sha256:" + ("cd" * 32)
_OTHER_DIGEST = "sha256:" + ("ef" * 32)
_IDEM = "idem-pg-fix-c"
_POLICY_RULE = "runtime.hitl"
_BUNDLE_ID = "bundle-pg-c"
_BUNDLE_V1 = "1.0.0"
_BUNDLE_V2 = "2.0.0"
_BUNDLE_D1 = "sha256:" + ("11" * 32)
_BUNDLE_D2 = "sha256:" + ("22" * 32)
_SCOPE = "external_work.mutate"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


class MutableRuntimePolicyEvaluator:
    def __init__(self, decision: PolicyDecision) -> None:
        self._decision = decision
        self.calls: list[MeaningfulSideEffectRequest] = []

    def set_decision(self, decision: PolicyDecision) -> None:
        self._decision = decision

    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        self.calls.append(request)
        return self._decision


def _decision(
    *,
    action: PolicyAction = PolicyAction.REQUIRE_HUMAN,
    policy_rule_id: str = _POLICY_RULE,
    policy_bundle_version: str = _BUNDLE_V1,
    policy_bundle_digest: str = _BUNDLE_D1,
) -> PolicyDecision:
    return PolicyDecision(
        action=action,
        reason="pg-fix-c-test",
        policy_rule_id=policy_rule_id,
        policy_bundle_id=_BUNDLE_ID,
        policy_bundle_version=policy_bundle_version,
        policy_bundle_digest=policy_bundle_digest,
    )


def _grant(**overrides: object) -> GovernedContinuationApprovalGrant:
    payload: dict[str, object] = {
        "grant_id": "gcg_pg_fix_c",
        "continuation_request_id": "gcr_pg_fix_c",
        "side_effect_scope_id": _IDEM,
        "side_effect_scope_digest": _DIGEST,
        "task_id": _TASK_ID,
        "run_id": _RUN_ID,
        "operation_id": ACTION_CREATE_EXTERNAL_WORK,
        "resource_scope": _DIGEST,
        "policy_rule_id": _POLICY_RULE,
        "policy_bundle_id": _BUNDLE_ID,
        "policy_bundle_version": _BUNDLE_V1,
        "policy_bundle_digest": _BUNDLE_D1,
        "pause_id": "pause-pg-c",
        "human_request_id": "hr-pg-c",
        "approved_at": "2026-08-19T00:00:00+00:00",
    }
    payload.update(overrides)
    return GovernedContinuationApprovalGrant.model_validate(payload)


def _meta() -> dict[str, object]:
    return {
        META_PROVIDER_ID: "gec3_deterministic_fake",
        META_SCOPE_DESCRIPTION: "review PR #42",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: _IDEM,
        META_WORKSPACE_REF: _WORKSPACE,
        "external_work.budget_limit": MoneyAmount(amount=Decimal("40.00"), currency="USD"),
        "external_work.principal_id": _PRINCIPAL,
        "external_work.tenant_id": _TENANT,
    }


class _RecordingIntegration(DeterministicExternalWorkFake):
    def __init__(self, *, call_log: list[str] | None = None) -> None:
        super().__init__()
        self.call_log = call_log if call_log is not None else []
        self.grant_probe: object | None = None

    def create_work(self, request):  # type: ignore[no-untyped-def]
        if self.grant_probe is not None:
            grant = self.grant_probe()
            self.call_log.append(f"grant_present={grant is not None}")
        self.call_log.append("integration.create_work")
        return super().create_work(request)


def _seed_boundary(
    evaluator: object,
) -> MeaningfulSideEffectAuthorizationBoundary:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-pg-c",
            principal_id=_PRINCIPAL,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-pg-c",
            principal_id=_PRINCIPAL,
            authority_scopes=(_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    profile_repo.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation_id=ACTION_CREATE_EXTERNAL_WORK,
            authority_scope=_SCOPE,
            workspace_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
            status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
        )
    )
    gate = CollaborativeWorkEnforcementGate(
        profile_repository=profile_repo,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=InMemoryAuthorityDelegationRepository(),
            principal_authority_repository=authority_repo,
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(policy_repo),
        runtime_policy_evaluator=evaluator,
    )
    return MeaningfulSideEffectAuthorizationBoundary(enforcement_gate=gate)


def _task(task_id: str = _TASK_ID) -> Task:
    return Task(tenant_id=_TENANT, user_id=_PRINCIPAL, message="x", task_id=task_id)


def _adapter_with_grant(
    *,
    evaluator: MutableRuntimePolicyEvaluator | None = None,
    call_log: list[str] | None = None,
    task: Task | None = None,
) -> tuple[ExternalWorkAdapter, Task, list[str]]:
    runtime = evaluator or MutableRuntimePolicyEvaluator(_decision())
    boundary = _seed_boundary(runtime)
    log: list[str] = call_log if call_log is not None else []
    integration = _RecordingIntegration(call_log=log)
    owned_task = task or _task()
    integration.grant_probe = lambda: owned_task.runtime.governance.governed_continuation_grant
    adapter = ExternalWorkAdapter(integration, authorization_boundary=boundary)
    owned_task.runtime.governance.governed_continuation_grant = _grant()
    return adapter, owned_task, log


def _create(
    adapter: ExternalWorkAdapter,
    task: Task,
    *,
    digest: str = _DIGEST,
    idem: str = _IDEM,
) -> object:
    meta = _meta()
    meta[META_SCOPE_DIGEST] = digest
    meta[META_IDEMPOTENCY_KEY] = idem
    request = adapter.build_create_request(
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        metadata=meta,
    )
    return adapter.create_and_map(
        request,
        enrich=False,
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
        task=task,
    )


def test_c1_matching_exact_grant_executes_once() -> None:
    adapter, task, log = _adapter_with_grant()
    result = _create(adapter, task)
    assert result.used is True
    assert log == ["grant_present=False", "integration.create_work"]
    assert task.runtime.governance.governed_continuation_grant is None


def test_c2_single_use_requires_fresh_approval() -> None:
    adapter, task, log = _adapter_with_grant()
    first = _create(adapter, task)
    assert first.used is True
    log.clear()
    second = _create(adapter, task)
    assert second.used is False
    assert second.reason == "side_effect_governance_required"
    assert log == []


def test_c3_wrong_task_blocks_provider() -> None:
    adapter, task, log = _adapter_with_grant()
    task.runtime.governance.governed_continuation_grant = _grant(task_id=_OTHER_TASK)
    result = _create(adapter, task)
    assert result.used is False
    assert log == []


def test_c4_wrong_run_blocks_provider() -> None:
    adapter, task, log = _adapter_with_grant()
    task.runtime.governance.governed_continuation_grant = _grant(run_id=_OTHER_RUN)
    result = _create(adapter, task)
    assert result.used is False
    assert log == []


def test_c5_wrong_operation_blocks_provider() -> None:
    adapter, task, log = _adapter_with_grant()
    task.runtime.governance.governed_continuation_grant = _grant(operation_id="UPDATE_EXTERNAL_WORK")
    result = _create(adapter, task)
    assert result.used is False
    assert log == []


def test_c6_wrong_resource_blocks_provider() -> None:
    adapter, task, log = _adapter_with_grant()
    task.runtime.governance.governed_continuation_grant = _grant(resource_scope=_OTHER_DIGEST)
    result = _create(adapter, task)
    assert result.used is False
    assert log == []


def test_c7_wrong_scope_id_blocks_provider() -> None:
    adapter, task, log = _adapter_with_grant()
    task.runtime.governance.governed_continuation_grant = _grant(side_effect_scope_id="other-idem")
    result = _create(adapter, task)
    assert result.used is False
    assert log == []


def test_c8_wrong_scope_digest_blocks_provider() -> None:
    adapter, task, log = _adapter_with_grant()
    task.runtime.governance.governed_continuation_grant = _grant(side_effect_scope_digest=_OTHER_DIGEST)
    result = _create(adapter, task)
    assert result.used is False
    assert log == []


def test_c9_wrong_policy_rule_blocks_provider() -> None:
    adapter, task, log = _adapter_with_grant()
    task.runtime.governance.governed_continuation_grant = _grant(policy_rule_id="other.rule")
    result = _create(adapter, task)
    assert result.used is False
    assert log == []


def test_c10_wrong_policy_bundle_blocks_provider() -> None:
    adapter, task, log = _adapter_with_grant()
    task.runtime.governance.governed_continuation_grant = _grant(
        policy_bundle_version=_BUNDLE_V2,
        policy_bundle_digest=_BUNDLE_D2,
    )
    result = _create(adapter, task)
    assert result.used is False
    assert log == []


def test_c11_fresh_deny_over_grant_blocks_provider() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision(action=PolicyAction.DENY))
    adapter, task, log = _adapter_with_grant(evaluator=evaluator)
    result = _create(adapter, task)
    assert result.used is False
    assert result.reason == "side_effect_denied"
    assert log == []
    assert task.runtime.governance.governed_continuation_grant is None


def test_c12_fresh_allow_clears_stale_grant() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision(action=PolicyAction.ALLOW))
    adapter, task, log = _adapter_with_grant(evaluator=evaluator)
    result = _create(adapter, task)
    assert result.used is True
    assert log == ["grant_present=False", "integration.create_work"]
    assert task.runtime.governance.governed_continuation_grant is None


def test_c13_provider_failure_leaves_grant_consumed() -> None:
    class _FailingIntegration(_RecordingIntegration):
        def create_work(self, request):  # type: ignore[no-untyped-def]
            self.call_log.append("integration.create_work")
            raise RuntimeError("provider_failed")

    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary = _seed_boundary(evaluator)
    log: list[str] = []
    integration = _FailingIntegration(call_log=log)
    task = _task()
    adapter = ExternalWorkAdapter(integration, authorization_boundary=boundary)
    task.runtime.governance.governed_continuation_grant = _grant()
    result = _create(adapter, task)
    assert result.used is False
    assert result.reason == "side_effect_authorization_failed"
    assert task.runtime.governance.governed_continuation_grant is None


def test_c14_grant_consumed_before_provider_callback() -> None:
    adapter, task, log = _adapter_with_grant()
    _create(adapter, task)
    assert log[0] == "grant_present=False"


def test_c15_approval_provenance_still_required() -> None:
    continuation = GovernedContinuationRequest(
        reason=ContinuationReason.COMPLIANCE,
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        source_agent_id="agent-test",
        prompt="continuation required",
        continuation_request_id="gcr_pg_c15",
        side_effect_scope_id=_IDEM,
        side_effect_scope_digest=_DIGEST,
        operation_id=ACTION_CREATE_EXTERNAL_WORK,
        policy_rule_id=_POLICY_RULE,
        policy_bundle_id=_BUNDLE_ID,
        policy_bundle_version=_BUNDLE_V1,
        policy_bundle_digest=_BUNDLE_D1,
        resource_scope=_DIGEST,
        policy_action=PolicyAction.REQUIRE_HUMAN,
    )
    task = _task()
    HumanPauseCoordinator.apply_pause(
        task,
        bridge_governed_continuation_to_execution_result(continuation),
    )
    pause = task.runtime.governance.pause_record
    assert pause is not None
    HumanPauseCoordinator.resolve_human_response(
        task,
        HumanResponseVerdict.APPROVE,
        approver=local_development_approver_evidence(tenant_id=_TENANT),
        pause_id=pause.pause_id,
        human_request_id=pause.human_request_id,
        run_id=_RUN_ID,
    )
    resolution = task.runtime.governance.hitl_resolution
    assert resolution is not None
    task.runtime.governance.hitl_resolution = resolution.model_copy(
        update={"pause_id": "pause-wrong"}
    )
    with pytest.raises(GovernedContinuationGrantError):
        GovernedContinuationGrantCoordinator.create_grant_from_approval(task)


def test_c16_pg_fix_b_regression_deny_over_grant() -> None:
    from intergrax.contracts.meaningful_side_effect_policy import MeaningfulSideEffectPolicyRule
    from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

    engine = RuntimePolicyEngine(
        meaningful_side_effect_rules=(
            MeaningfulSideEffectPolicyRule(
                rule_id="pgc.wildcard.allow",
                decision=PolicyAction.ALLOW,
            ),
            MeaningfulSideEffectPolicyRule(
                rule_id="pgc.specific.deny",
                action=ACTION_CREATE_EXTERNAL_WORK,
                decision=PolicyAction.DENY,
            ),
        )
    )
    adapter, task, log = _adapter_with_grant(evaluator=engine)
    result = _create(adapter, task)
    assert result.used is False
    assert result.policy_decision is not None
    assert result.policy_decision.action is PolicyAction.DENY
    assert log == []


def test_c17_pg_fix_a_authority_regression() -> None:
    boundary = seed_external_work_authorization_boundary(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        seed_workspace_policy=False,
    )
    adapter = ExternalWorkAdapter(DeterministicExternalWorkFake(), authorization_boundary=boundary)
    task = _task()
    task.runtime.governance.governed_continuation_grant = _grant()
    result = _create(adapter, task)
    assert result.used is False
    assert result.reason == "side_effect_denied"


def test_c18_checkpoint_resume_grant_survives_and_consumes() -> None:
    task = _task()
    task.runtime.governance.governed_continuation_grant = _grant()
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        resume_token="rt-pg-c",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
    )
    restored = Task.model_validate(checkpoint.task_snapshot)
    assert restored.runtime.governance.governed_continuation_grant is not None
    adapter, _, log = _adapter_with_grant(task=restored)
    result = _create(adapter, restored)
    assert result.used is True
    assert restored.runtime.governance.governed_continuation_grant is None
    assert "integration.create_work" in log
