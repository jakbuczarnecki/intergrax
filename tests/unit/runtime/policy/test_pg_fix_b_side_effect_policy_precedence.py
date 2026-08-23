# © Artur Czarnecki. All rights reserved.

"""PG-FIX-B — deterministic meaningful-side-effect policy precedence."""

from __future__ import annotations

import itertools
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
    CreateCollaborativePolicyRuleCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativePolicyRuleStatus,
    CollaborativeWorkEnforcementRequest,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.meaningful_side_effect_policy import MeaningfulSideEffectPolicyRule
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
    MeaningfulSideEffectAuthorizationResult,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.task.task import Task

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_ACTION = "DELETE_EXTERNAL_WORK"
_DIGEST = "sha256:" + ("cd" * 32)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_TASK = "task-pg-b"
_RUN = "run-pg-b"
_PRINCIPAL = "principal-pg-b"
_SCOPE = "external_work.mutate"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


def _request(**overrides: object) -> MeaningfulSideEffectRequest:
    payload: dict[str, object] = {
        "action": _ACTION,
        "kinds": (MeaningfulSideEffectKind.MUTATION,),
        "side_effect_scope_id": "scope-pg-b",
        "task_id": _TASK,
        "run_id": _RUN,
        "principal_id": _PRINCIPAL,
        "tenant_id": _TENANT,
    }
    payload.update(overrides)
    return MeaningfulSideEffectRequest.model_validate(payload)


def _rule(**overrides: object) -> MeaningfulSideEffectPolicyRule:
    payload: dict[str, object] = {
        "rule_id": "pgb.rule",
        "decision": PolicyAction.ALLOW,
        "action": _ACTION,
    }
    payload.update(overrides)
    return MeaningfulSideEffectPolicyRule(**payload)  # type: ignore[arg-type]


def _engine(rules: tuple[MeaningfulSideEffectPolicyRule, ...]) -> RuntimePolicyEngine:
    return RuntimePolicyEngine(meaningful_side_effect_rules=rules)


def _wildcard_allow() -> MeaningfulSideEffectPolicyRule:
    return MeaningfulSideEffectPolicyRule(
        rule_id="pgb.wildcard.allow",
        decision=PolicyAction.ALLOW,
    )


def _specific_deny() -> MeaningfulSideEffectPolicyRule:
    return MeaningfulSideEffectPolicyRule(
        rule_id="pgb.specific.deny",
        action=_ACTION,
        decision=PolicyAction.DENY,
    )


def test_b1_historical_exploit_wildcard_allow_specific_deny() -> None:
    decision = _engine((_wildcard_allow(), _specific_deny())).evaluate_meaningful_side_effect(
        _request(),
    )
    assert decision.action is PolicyAction.DENY
    assert decision.policy_rule_id == "pgb.specific.deny"


def test_b2_reverse_order_same_deny() -> None:
    decision = _engine((_specific_deny(), _wildcard_allow())).evaluate_meaningful_side_effect(
        _request(),
    )
    assert decision.action is PolicyAction.DENY
    assert decision.policy_rule_id == "pgb.specific.deny"


def test_b3_permutation_invariance() -> None:
    rules = (
        _wildcard_allow(),
        _specific_deny(),
        MeaningfulSideEffectPolicyRule(
            rule_id="pgb.wildcard.hitl",
            decision=PolicyAction.REQUIRE_HUMAN,
        ),
        MeaningfulSideEffectPolicyRule(
            rule_id="pgb.specific.escalate",
            action=_ACTION,
            decision=PolicyAction.ESCALATE,
        ),
    )
    expected = _engine(rules).evaluate_meaningful_side_effect(_request()).action
    for permutation in itertools.permutations(rules):
        engine = _engine(permutation)
        assert engine.evaluate_meaningful_side_effect(_request()).action is expected


def test_b4_wildcard_allow_specific_require_human() -> None:
    decision = _engine(
        (
            _wildcard_allow(),
            MeaningfulSideEffectPolicyRule(
                rule_id="pgb.specific.hitl",
                action=_ACTION,
                decision=PolicyAction.REQUIRE_HUMAN,
            ),
        )
    ).evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.REQUIRE_HUMAN
    assert decision.policy_rule_id == "pgb.specific.hitl"


def test_b5_global_deny_blocks_specific_allow() -> None:
    decision = _engine(
        (
            MeaningfulSideEffectPolicyRule(
                rule_id="pgb.wildcard.deny",
                decision=PolicyAction.DENY,
            ),
            MeaningfulSideEffectPolicyRule(
                rule_id="pgb.specific.allow",
                action=_ACTION,
                decision=PolicyAction.ALLOW,
            ),
        )
    ).evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.DENY
    assert decision.policy_rule_id == "pgb.wildcard.deny"


def test_b6_deny_dominates_require_human() -> None:
    decision = _engine(
        (
            MeaningfulSideEffectPolicyRule(
                rule_id="pgb.specific.hitl",
                action=_ACTION,
                decision=PolicyAction.REQUIRE_HUMAN,
            ),
            _specific_deny(),
        )
    ).evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.DENY
    assert decision.policy_rule_id == "pgb.specific.deny"


def test_b7_duplicate_same_specificity_allow_and_deny() -> None:
    decision = _engine(
        (
            MeaningfulSideEffectPolicyRule(
                rule_id="pgb.specific.allow",
                action=_ACTION,
                decision=PolicyAction.ALLOW,
            ),
            MeaningfulSideEffectPolicyRule(
                rule_id="pgb.specific.deny.alt",
                action=_ACTION,
                decision=PolicyAction.DENY,
            ),
        )
    ).evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.DENY
    assert decision.policy_rule_id == "pgb.specific.deny.alt"


def test_b8_modify_unsupported_conservative_deny() -> None:
    decision = _engine(
        (
            _wildcard_allow(),
            MeaningfulSideEffectPolicyRule(
                rule_id="pgb.specific.modify",
                action=_ACTION,
                decision=PolicyAction.MODIFY,
            ),
        )
    ).evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "meaningful_side_effect_unsupported_decision"
    assert decision.policy_rule_id == "pgb.specific.modify"


def test_b9_no_match_indeterminate_deny() -> None:
    decision = _engine(
        (
            MeaningfulSideEffectPolicyRule(
                rule_id="pgb.other",
                action="OTHER_ACTION",
                decision=PolicyAction.ALLOW,
            ),
        )
    ).evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "meaningful_side_effect_indeterminate"


def test_b10_identity_fail_closed_regression() -> None:
    engine = _engine((_rule(),))
    identity_decision = engine.evaluate_meaningful_side_effect(
        MeaningfulSideEffectRequest.model_construct(
            action=_ACTION,
            kinds=(MeaningfulSideEffectKind.MUTATION,),
            side_effect_scope_id="scope-pg-b",
            task_id="",
            run_id=_RUN,
            principal_id=_PRINCIPAL,
        ),
    )
    assert identity_decision.action is PolicyAction.DENY
    assert identity_decision.reason == "meaningful_side_effect_identity_missing"

    principal_decision = engine.evaluate_meaningful_side_effect(_request(principal_id=None))
    assert principal_decision.action is PolicyAction.DENY
    assert principal_decision.reason == "meaningful_side_effect_principal_missing"


def test_b11_controlling_rule_evidence_reports_restrictive_rule() -> None:
    decision = _engine((_wildcard_allow(), _specific_deny())).evaluate_meaningful_side_effect(
        _request(),
    )
    assert decision.policy_rule_id == "pgb.specific.deny"
    assert "pgb.wildcard.allow" in decision.audit_payload.get("matched_rule_ids", [])
    assert decision.audit_payload.get("resolution_reason") == "conservative_precedence"


def test_wildcard_require_human_blocks_specific_allow() -> None:
    decision = _engine(
        (
            MeaningfulSideEffectPolicyRule(
                rule_id="pgb.wildcard.hitl",
                decision=PolicyAction.REQUIRE_HUMAN,
            ),
            MeaningfulSideEffectPolicyRule(
                rule_id="pgb.specific.allow",
                action=_ACTION,
                decision=PolicyAction.ALLOW,
            ),
        )
    ).evaluate_meaningful_side_effect(_request())
    assert decision.action is PolicyAction.REQUIRE_HUMAN
    assert decision.policy_rule_id == "pgb.wildcard.hitl"


def test_no_first_match_break_in_source() -> None:
    import inspect

    source = inspect.getsource(RuntimePolicyEngine.evaluate_meaningful_side_effect)
    assert "break" not in source


class _RecordingIntegration(DeterministicExternalWorkFake):
    def __init__(self, *, call_log: list[str] | None = None) -> None:
        super().__init__()
        self.call_log = call_log if call_log is not None else []

    def create_work(self, request):  # type: ignore[no-untyped-def]
        self.call_log.append("integration.create_work")
        return super().create_work(request)


def _meta(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        META_PROVIDER_ID: "gec3_deterministic_fake",
        META_SCOPE_DESCRIPTION: "review PR #42",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: "idem-pg-b",
        META_WORKSPACE_REF: _WORKSPACE,
        "external_work.budget_limit": MoneyAmount(amount=Decimal("40.00"), currency="USD"),
        "external_work.principal_id": _PRINCIPAL,
        "external_work.tenant_id": _TENANT,
    }
    payload.update(overrides)
    return payload


def _seed_external_work_boundary(
    runtime_engine: RuntimePolicyEngine,
) -> MeaningfulSideEffectAuthorizationBoundary:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()

    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-pg-b",
            principal_id=_PRINCIPAL,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-pg-b",
            principal_id=_PRINCIPAL,
            authority_scopes=(_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="workspace-allow-pg-b",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_SCOPE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    profile_repo.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation_id=ACTION_CREATE_EXTERNAL_WORK,
            authority_scope=_SCOPE,
            workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
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
        runtime_policy_evaluator=runtime_engine,
    )
    return MeaningfulSideEffectAuthorizationBoundary(enforcement_gate=gate)


def test_b12_external_work_integration_broad_allow_cannot_bypass_specific_deny() -> None:
    runtime = _engine(
        (
            MeaningfulSideEffectPolicyRule(
                rule_id="pgb.integration.wildcard.allow",
                decision=PolicyAction.ALLOW,
            ),
            MeaningfulSideEffectPolicyRule(
                rule_id="pgb.integration.specific.deny",
                action=ACTION_CREATE_EXTERNAL_WORK,
                decision=PolicyAction.DENY,
            ),
        )
    )
    boundary = _seed_external_work_boundary(runtime)
    call_log: list[str] = []
    adapter = ExternalWorkAdapter(
        _RecordingIntegration(call_log=call_log),
        authorization_boundary=boundary,
    )
    denied = adapter.create_and_map(
        adapter.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta()),
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    assert denied.used is False
    assert denied.policy_decision is not None
    assert denied.policy_decision.action is PolicyAction.DENY
    assert call_log == []


_TASK_ID = mint_task_id()
_RUN_ID = "run-pg-b-grant"
_OPERATION = ACTION_CREATE_EXTERNAL_WORK
_BUNDLE_ID = "bundle-pg-b"
_BUNDLE_V1 = "1.0.0"
_BUNDLE_D1 = "sha256:" + ("11" * 32)
_POLICY_RULE = "pgb.integration.specific.deny"


def _grant() -> GovernedContinuationApprovalGrant:
    return GovernedContinuationApprovalGrant(
        grant_id="gcg_pg_b_test",
        continuation_request_id="gcr_pg_b_test",
        side_effect_scope_id="scope-pg-b",
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        operation_id=_OPERATION,
        resource_scope=None,
        policy_rule_id=_POLICY_RULE,
        policy_bundle_id=_BUNDLE_ID,
        policy_bundle_version=_BUNDLE_V1,
        policy_bundle_digest=_BUNDLE_D1,
        pause_id="pause-pg-b",
        human_request_id="hr-pg-b",
        approved_at="2026-08-22T00:00:00+00:00",
    )


def _seed_grant_boundary(
    runtime_engine: RuntimePolicyEngine,
) -> tuple[MeaningfulSideEffectAuthorizationBoundary, WorkspaceMembership, Task]:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()

    membership = membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-pg-b-grant",
            principal_id=_PRINCIPAL,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-pg-b-grant",
            principal_id=_PRINCIPAL,
            authority_scopes=(_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="workspace-allow-pg-b-grant",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_SCOPE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    profile_repo.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation_id=_OPERATION,
            authority_scope=_SCOPE,
            workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
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
        runtime_policy_evaluator=runtime_engine,
    )
    boundary = MeaningfulSideEffectAuthorizationBoundary(enforcement_gate=gate)
    task = Task(tenant_id=_TENANT, user_id=_PRINCIPAL, message="x", task_id=_TASK_ID)
    return boundary, membership, task


def test_b13_deny_over_stored_grant_blocks_execution() -> None:
    runtime = _engine(
        (
            MeaningfulSideEffectPolicyRule(
                rule_id="pgb.integration.wildcard.allow",
                decision=PolicyAction.ALLOW,
            ),
            MeaningfulSideEffectPolicyRule(
                rule_id=_POLICY_RULE,
                action=_OPERATION,
                decision=PolicyAction.DENY,
            ),
        )
    )
    boundary, membership, task = _seed_grant_boundary(runtime)
    task.runtime.governance.governed_continuation_grant = _grant()
    counter = [0]

    def _execute() -> str:
        counter[0] += 1
        return "ok"

    request = CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=_OPERATION,
        acting_principal_id=_PRINCIPAL,
        membership=WorkspaceMembership.model_validate(membership.model_dump()),
        meaningful_side_effect_request=MeaningfulSideEffectRequest(
            action=_OPERATION,
            kinds=(MeaningfulSideEffectKind.MUTATION,),
            side_effect_scope_id="scope-pg-b",
            task_id=_TASK_ID,
            run_id=_RUN_ID,
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
        ),
    )
    result = boundary.authorize_and_execute(
        request,
        _execute,
        task=task,
    )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert counter[0] == 0
