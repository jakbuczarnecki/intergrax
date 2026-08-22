# © Artur Czarnecki. All rights reserved.

"""PG-FIX-A — authoritative membership locator proofs for External Work."""

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
from external_contractor_adapter.tests.fakes.deterministic_side_effect_policy import (
    DeterministicMeaningfulSideEffectPolicy,
)
from external_contractor_adapter.tests.fakes.external_work_authorization_boundary import (
    seed_external_work_authorization_boundary,
    workspace_membership_locator,
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
    EffectiveAuthorityDenialReason,
    EffectiveAuthorityRequest,
    MembershipResolutionMode,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    WorkspaceMembershipRole,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectKind, MeaningfulSideEffectRequest
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_DIGEST = "sha256:" + ("cd" * 32)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_TASK = "task-auth-membership"
_RUN = "run-auth-membership"
_PRINCIPAL = "principal-pg-a1"
_SCOPE = "external_work.mutate"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


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
        META_IDEMPOTENCY_KEY: "idem-auth-membership",
        META_WORKSPACE_REF: _WORKSPACE,
        "external_work.budget_limit": MoneyAmount(amount=Decimal("40.00"), currency="USD"),
        "external_work.principal_id": _PRINCIPAL,
        "external_work.tenant_id": _TENANT,
    }
    payload.update(overrides)
    return payload


def _seed_gate(
    *,
    membership_id: str = "real-membership-123",
    membership_status: MembershipStatus = MembershipStatus.ACTIVE,
    principal_id: str = _PRINCIPAL,
    seed_authority: bool = True,
    authority_scopes: tuple[str, ...] = (_SCOPE,),
    seed_resource_policy: bool = False,
    resource_deny_scopes: tuple[str, ...] = (),
    resource_allow_scopes: tuple[str, ...] = (),
) -> tuple[MeaningfulSideEffectAuthorizationBoundary, InMemoryWorkspaceMembershipRepository]:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()

    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id=membership_id,
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=membership_status,
        )
    )
    if seed_authority:
        authority_repo.create(
            CreatePrincipalAuthorityGrantCommand(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                authority_grant_id=f"grant-{principal_id}",
                principal_id=principal_id,
                authority_scopes=authority_scopes,
                status=AuthorityGrantStatus.ACTIVE,
            )
        )

    policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="workspace-allow",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_SCOPE,
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    for index, resource_scope in enumerate(resource_allow_scopes):
        policy_repo.create(
            CreateCollaborativePolicyRuleCommand(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                policy_rule_id=f"resource-allow-{index}",
                layer=PolicyCompositionLayer.RESOURCE_POLICY,
                authority_scope=_SCOPE,
                resource_scope=resource_scope,
                action=PolicyAction.ALLOW,
                status=CollaborativePolicyRuleStatus.ACTIVE,
            )
        )
    for index, resource_scope in enumerate(resource_deny_scopes):
        policy_repo.create(
            CreateCollaborativePolicyRuleCommand(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                policy_rule_id=f"resource-deny-{index}",
                layer=PolicyCompositionLayer.RESOURCE_POLICY,
                authority_scope=_SCOPE,
                resource_scope=resource_scope,
                action=PolicyAction.DENY,
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
            resource_policy_applicability=(
                PolicyLayerApplicability.REQUIRED if seed_resource_policy else PolicyLayerApplicability.NOT_APPLICABLE
            ),
            runtime_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_requirement=(
                OperationPolicyRequirement.REQUIRED
                if seed_resource_policy
                else OperationPolicyRequirement.NOT_APPLICABLE
            ),
            meaningful_side_effect_requirement=OperationPolicyRequirement.REQUIRED,
            status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
        )
    )

    runtime = DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.ALLOW)
    gate = CollaborativeWorkEnforcementGate(
        profile_repository=profile_repo,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=InMemoryAuthorityDelegationRepository(),
            principal_authority_repository=authority_repo,
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(policy_repo),
        runtime_policy_evaluator=runtime,
    )
    return MeaningfulSideEffectAuthorizationBoundary(enforcement_gate=gate), membership_repo


def _adapter_call(
    boundary: MeaningfulSideEffectAuthorizationBoundary,
    *,
    principal_id: str,
    call_log: list[str],
    idempotency_key: str = "idem-auth-membership",
) -> object:
    adapter = ExternalWorkAdapter(_RecordingIntegration(call_log=call_log), authorization_boundary=boundary)
    return adapter.create_and_map(
        adapter.build_create_request(
            task_id=_TASK,
            run_id=_RUN,
            metadata=_meta(**{META_IDEMPOTENCY_KEY: idempotency_key}),
        ),
        principal_id=principal_id,
        tenant_id=_TENANT,
    )


def test_no_membership_denies_with_zero_provider_calls() -> None:
    call_log: list[str] = []
    boundary, _ = _seed_gate()
    denied = _adapter_call(boundary, principal_id="principal-without-membership", call_log=call_log)
    assert denied.used is False
    assert denied.reason == "side_effect_denied"
    assert call_log == []


def test_forged_membership_locator_denies_with_zero_provider_calls() -> None:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    resolver = CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo,
        delegation_repository=InMemoryAuthorityDelegationRepository(),
        principal_authority_repository=authority_repo,
        clock=lambda: _NOW,
    )
    forged = workspace_membership_locator(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id="principal-b",
    )
    decision = resolver.resolve(
        EffectiveAuthorityRequest(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            acting_principal_id="principal-b",
            requested_authority_scopes=(_SCOPE,),
            membership=forged,
            membership_resolution_mode=MembershipResolutionMode.LOCATOR,
        )
    )
    assert decision.decision.action is PolicyAction.DENY
    assert decision.denial_reason is EffectiveAuthorityDenialReason.MISSING_MEMBERSHIP


def test_wrong_membership_id_denies_then_canonical_allows() -> None:
    call_log_wrong: list[str] = []
    call_log_canonical: list[str] = []
    boundary, _ = _seed_gate(membership_id="real-membership-123")
    wrong_locator = workspace_membership_locator(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    wrong_request = CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=ACTION_CREATE_EXTERNAL_WORK,
        acting_principal_id=_PRINCIPAL,
        resource_scope=_DIGEST,
        membership=wrong_locator,
        membership_resolution_mode=MembershipResolutionMode.LOCATOR,
        meaningful_side_effect_request=MeaningfulSideEffectRequest(
            action=ACTION_CREATE_EXTERNAL_WORK,
            kinds=(MeaningfulSideEffectKind.MUTATION,),
            side_effect_scope_id="scope-wrong-id",
            task_id=_TASK,
            run_id=_RUN,
            principal_id=_PRINCIPAL,
            tenant_id=_TENANT,
            resource=_DIGEST,
        ),
    )
    wrong_result = boundary.authorize(wrong_request)
    assert wrong_result.permitted is False

    allowed = _adapter_call(boundary, principal_id=_PRINCIPAL, call_log=call_log_canonical, idempotency_key="idem-canonical")
    assert allowed.used is True
    assert call_log_wrong == []
    assert call_log_canonical.count("integration.create_work") == 1


def test_inactive_membership_denies_with_zero_provider_calls() -> None:
    for status in (MembershipStatus.SUSPENDED, MembershipStatus.REVOKED):
        call_log: list[str] = []
        boundary, _ = _seed_gate(membership_status=status)
        denied = _adapter_call(
            boundary,
            principal_id=_PRINCIPAL,
            call_log=call_log,
            idempotency_key=f"idem-{status.value}",
        )
        assert denied.used is False
        assert denied.reason == "side_effect_denied"
        assert call_log == []


def test_active_membership_without_base_authority_denies() -> None:
    call_log: list[str] = []
    boundary, _ = _seed_gate(seed_authority=False)
    denied = _adapter_call(boundary, principal_id=_PRINCIPAL, call_log=call_log, idempotency_key="idem-no-grant")
    assert denied.used is False
    assert denied.reason == "side_effect_denied"
    assert call_log == []


def test_insufficient_base_authority_scope_denies() -> None:
    call_log: list[str] = []
    boundary = seed_external_work_authorization_boundary(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        runtime_policy_evaluator=DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.ALLOW),
        extra_principal_grants={"weak-principal": ("external_work.read",)},
    )
    denied = _adapter_call(
        boundary,
        principal_id="weak-principal",
        call_log=call_log,
        idempotency_key="idem-weak-scope",
    )
    assert denied.used is False
    assert denied.reason == "side_effect_denied"
    assert call_log == []


def test_resource_scope_uses_repository_authority() -> None:
    other_digest = "sha256:" + ("ef" * 32)
    call_log_denied: list[str] = []
    call_log_allowed: list[str] = []
    boundary, _ = _seed_gate(
        seed_resource_policy=True,
        resource_deny_scopes=(_DIGEST,),
        resource_allow_scopes=(other_digest,),
    )
    denied = _adapter_call(
        boundary,
        principal_id=_PRINCIPAL,
        call_log=call_log_denied,
        idempotency_key="idem-resource-deny",
    )
    assert denied.used is False
    assert call_log_denied == []

    adapter = ExternalWorkAdapter(
        _RecordingIntegration(call_log=call_log_allowed),
        authorization_boundary=boundary,
    )
    allowed = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-resource-allow",
            run_id="run-resource-allow",
            metadata=_meta(**{META_SCOPE_DIGEST: other_digest, META_IDEMPOTENCY_KEY: "idem-resource-allow"}),
        ),
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    assert allowed.used is True
    assert call_log_allowed.count("integration.create_work") == 1
