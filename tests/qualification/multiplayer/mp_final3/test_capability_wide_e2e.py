# © Artur Czarnecki. All rights reserved.

"""MP-FINAL-3 — capability-wide backend E2E (happy path + negatives)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.collaborative_activity import CollaborativeActivityQuery
from intergrax.contracts.collaborative_activity_read import (
    CollaborativeActivityReadDenied,
    CollaborativeActivityReadRequest,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeWorkAuthorizationDenied,
    CreateWorkItemRequest,
    MembershipResolutionMode,
)
from intergrax.contracts.context_view_visibility_policy import ContextViewPolicyOutcome
from tests.qualification.multiplayer.mp_final3.host import (
    AUTHORIZED_PRINCIPAL,
    FOREIGN_PRINCIPAL,
    TENANT_A,
    TENANT_B,
    UNAUTHORIZED_PRINCIPAL,
    WS_A,
    WS_B,
    build_mp_final3_host,
)
from tests.qualification.multiplayer.mp_final3.scenario import (
    WORK_ITEM_ID,
    assert_happy_path_consistency,
    compose_context_view_for_principal,
    read_authorized_activity_page,
    run_capability_wide_happy_path,
)

pytestmark = pytest.mark.unit


def test_capability_wide_happy_path_e2e(tmp_path: Path) -> None:
    host = build_mp_final3_host(tmp_path / "mpf3-happy.sqlite")
    try:
        result = run_capability_wide_happy_path(host)
        assert_happy_path_consistency(result)

        loaded = host.decision_binding_service.get_binding(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            binding_id=result.binding.binding_id,
        )
        assert loaded is not None
        assert loaded.work_item_id == WORK_ITEM_ID
        assert loaded.decision_proposal.identity.decision_id == (
            result.decision_proposal.identity.decision_id
        )
    finally:
        host.close()


def test_unauthorized_mutation_denied(tmp_path: Path) -> None:
    host = build_mp_final3_host(tmp_path / "mpf3-deny.sqlite")
    try:
        with pytest.raises(CollaborativeWorkAuthorizationDenied):
            host.work_service.create_work_item(
                CreateWorkItemRequest(
                    tenant_id=TENANT_A,
                    workspace_id=WS_A,
                    work_item_id="mpf3-denied-wi",
                    acting_principal_id=UNAUTHORIZED_PRINCIPAL,
                    idempotency_key="mpf3-denied-create",
                    membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
                ),
            )
    finally:
        host.close()


def test_cross_tenant_activity_isolation(tmp_path: Path) -> None:
    host = build_mp_final3_host(tmp_path / "mpf3-iso-activity.sqlite")
    try:
        run_capability_wide_happy_path(host)
        # Tenant-B principal with B membership cannot read tenant-A activity stream.
        membership_b = host.bundle.membership.get_for_principal(
            tenant_id=TENANT_B,
            workspace_id=WS_B,
            principal_id=FOREIGN_PRINCIPAL,
        )
        assert membership_b is not None
        with pytest.raises(CollaborativeActivityReadDenied):
            host.activity_read_service.read_page(
                CollaborativeActivityReadRequest(
                    query=CollaborativeActivityQuery(
                        tenant_id=TENANT_A,
                        workspace_id=WS_A,
                        limit=50,
                    ),
                    acting_principal_id=FOREIGN_PRINCIPAL,
                    membership_resolution_mode=MembershipResolutionMode.LOCATOR,
                    membership=membership_b,
                ),
            )
        # Authorized work actor lacks activity-read scope → DENY before provider.
        membership_a = host.bundle.membership.get_for_principal(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            principal_id=AUTHORIZED_PRINCIPAL,
        )
        assert membership_a is not None
        with pytest.raises(CollaborativeActivityReadDenied):
            host.activity_read_service.read_page(
                CollaborativeActivityReadRequest(
                    query=CollaborativeActivityQuery(
                        tenant_id=TENANT_A,
                        workspace_id=WS_A,
                        limit=50,
                    ),
                    acting_principal_id=AUTHORIZED_PRINCIPAL,
                    membership_resolution_mode=MembershipResolutionMode.LOCATOR,
                    membership=membership_a,
                ),
            )
        page_b = read_authorized_activity_page(
            host,
            tenant_id=TENANT_B,
            workspace_id=WS_B,
            acting_principal_id=FOREIGN_PRINCIPAL,
        )
        assert page_b.activities == ()
    finally:
        host.close()


def test_context_view_isolation(tmp_path: Path) -> None:
    host = build_mp_final3_host(tmp_path / "mpf3-iso-cv.sqlite")
    try:
        run_capability_wide_happy_path(host)
        # Foreign tenant principal must not obtain ALLOW for tenant-A collaborative refs.
        outcome, view = compose_context_view_for_principal(
            host,
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            acting_principal_id=FOREIGN_PRINCIPAL,
            work_item_id=WORK_ITEM_ID,
        )
        assert outcome is ContextViewPolicyOutcome.DENY
        assert view is None

        # Same principal in own tenant/workspace — ALLOW, but no A refs.
        outcome_b, view_b = compose_context_view_for_principal(
            host,
            tenant_id=TENANT_B,
            workspace_id=WS_B,
            acting_principal_id=FOREIGN_PRINCIPAL,
            publish_activity=False,
        )
        assert outcome_b is ContextViewPolicyOutcome.ALLOW
        assert view_b is not None
        assert view_b.entries == ()
    finally:
        host.close()


def test_authorized_principal_is_not_implicit_owner_authority(tmp_path: Path) -> None:
    """Membership alone does not authorize; authorized principal uses explicit grant."""
    host = build_mp_final3_host(tmp_path / "mpf3-auth-inv.sqlite")
    try:
        membership = host.bundle.membership.get_for_principal(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            principal_id=UNAUTHORIZED_PRINCIPAL,
        )
        assert membership is not None
        grant = host.bundle.principal_authority.get_for_principal(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            principal_id=UNAUTHORIZED_PRINCIPAL,
        )
        assert grant is None
        auth_grant = host.bundle.principal_authority.get_for_principal(
            tenant_id=TENANT_A,
            workspace_id=WS_A,
            principal_id=AUTHORIZED_PRINCIPAL,
        )
        assert auth_grant is not None
        assert AUTHORIZED_PRINCIPAL != "owner"
    finally:
        host.close()
