# © Artur Czarnecki. All rights reserved.

"""Focused PostgreSQL integration test for Collaborative Work service (COLLAB-WORK-2E)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositoriesWithSharedWork
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    CreateCollaborativeOperationPolicyProfileCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    INITIAL_RECORD_REVISION,
)
from intergrax.collaborative_work.service import (
    CollaborativeWorkService,
    TRUSTED_OPERATION_WORK_ITEM_CREATE,
    TRUSTED_OPERATION_WORK_ITEM_TRANSITION,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfileStatus,
    CreateWorkItemRequest,
    MembershipResolutionMode,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
    TransitionWorkItemRequest,
    WorkItemState,
    WorkspaceMembershipRole,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

pytestmark = [pytest.mark.integration, pytest.mark.network]

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_ACTING = "principal-acting"
_AUTHORITY_SCOPE = "collaborative_work.manage"
_WORK_ITEM_ID = "work-item-postgresql"
_NOW = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)
_LATER = _NOW + timedelta(minutes=5)


class _UnusedRuntimeEvaluator:
    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        _ = request
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="runtime evaluator must not run for internal shared-work mutations",
            policy_rule_id="test.unexpected_runtime",
        )


def _seed_postgresql_bundle(
    bundle: CollaborativeWorkRepositoriesWithSharedWork,
) -> CollaborativeWorkService:
    bundle.membership.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-acting",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )
    bundle.principal_authority.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-acting",
            principal_id=_ACTING,
            authority_scopes=(_AUTHORITY_SCOPE,),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    for operation_id in (
        TRUSTED_OPERATION_WORK_ITEM_CREATE,
        TRUSTED_OPERATION_WORK_ITEM_TRANSITION,
    ):
        bundle.operation_profile.create(
            CreateCollaborativeOperationPolicyProfileCommand(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                operation_id=operation_id,
                authority_scope=_AUTHORITY_SCOPE,
                workspace_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
                meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
                status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
            )
        )
    gate = CollaborativeWorkEnforcementGate(
        profile_repository=bundle.operation_profile,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=bundle.membership,
            delegation_repository=bundle.delegation,
            principal_authority_repository=bundle.principal_authority,
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(bundle.policy),
        runtime_policy_evaluator=_UnusedRuntimeEvaluator(),
    )
    return CollaborativeWorkService(
        work_item_repository=bundle.work_item,
        assignment_repository=bundle.assignment,
        enforcement_gate=gate,
        clock=lambda: _LATER,
    )


def test_postgresql_service_work_item_create_and_transition(
    postgresql_collaborative_work_bundle: CollaborativeWorkRepositoriesWithSharedWork,
) -> None:
    bundle = postgresql_collaborative_work_bundle
    service = _seed_postgresql_bundle(bundle)
    created = service.create_work_item(
        CreateWorkItemRequest(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
            acting_principal_id=_ACTING,
            idempotency_key="postgresql-create",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        )
    )
    assert created.revision == INITIAL_RECORD_REVISION
    assert created.state is WorkItemState.OPEN

    transitioned = service.transition_work_item(
        TransitionWorkItemRequest(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
            expected_revision=created.revision,
            target_state=WorkItemState.ACTIVE,
            acting_principal_id=_ACTING,
            idempotency_key="postgresql-transition",
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
        )
    )
    assert transitioned.state is WorkItemState.ACTIVE
    assert transitioned.revision == created.revision + 1

    loaded = bundle.work_item.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id=_WORK_ITEM_ID,
    )
    assert loaded == transitioned
