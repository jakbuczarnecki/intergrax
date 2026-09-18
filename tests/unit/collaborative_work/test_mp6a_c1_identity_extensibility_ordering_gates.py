# © Artur Czarnecki. All rights reserved.

"""MP-6A-C1 — identity, extensibility, and timeline semantics architecture gates."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from intergrax.contracts.collaborative_activity import (
    ActivityIdempotencyKey,
    CollaborativeActivity,
    CollaborativeActivityActorRef,
    CollaborativeActivityBuiltinSource,
    CollaborativeActivityBuiltinType,
    CollaborativeActivityOutcome,
    CollaborativeActivityOutcomeStatus,
    CollaborativeActivityPublication,
    CollaborativeActivityQuery,
    CollaborativeActivityScope,
    CollaborativeActivitySourceId,
    CollaborativeActivityTypeId,
    WorkItemActivityTargetRef,
    mint_collaborative_activity_id,
)
from intergrax.contracts.collaborative_work import PrincipalKind

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)


def _actor(tenant: str = "tenant-a") -> CollaborativeActivityActorRef:
    return CollaborativeActivityActorRef(
        tenant_id=tenant,
        principal_id="principal-1",
        principal_kind=PrincipalKind.HUMAN,
    )


def _scope(
    tenant: str = "tenant-a",
    workspace: str = "ws-a",
    work_item: str = "wi-1",
) -> CollaborativeActivityScope:
    return CollaborativeActivityScope(
        tenant_id=tenant,
        workspace_id=workspace,
        work_item_id=work_item,
    )


def _key(
    *,
    tenant: str = "tenant-a",
    workspace: str = "ws-a",
    source_stable_id: str = "stable-1",
    source: CollaborativeActivitySourceId | None = None,
    activity_type: CollaborativeActivityTypeId | None = None,
) -> ActivityIdempotencyKey:
    return ActivityIdempotencyKey(
        tenant_id=tenant,
        workspace_id=workspace,
        source=source or CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
        source_stable_id=source_stable_id,
        activity_type=activity_type or CollaborativeActivityBuiltinType.WORK_ITEM_CREATED,
    )


def _publication(
    *,
    tenant: str = "tenant-a",
    workspace: str = "ws-a",
    source_stable_id: str = "stable-1",
) -> CollaborativeActivityPublication:
    return CollaborativeActivityPublication(
        idempotency_key=_key(
            tenant=tenant,
            workspace=workspace,
            source_stable_id=source_stable_id,
        ),
        actor=_actor(tenant),
        scope=_scope(tenant, workspace),
        target=WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_NOW,
    )


def test_mp6a_c1_cross_tenant_activity_id_differs() -> None:
    pub_a = _publication(tenant="tenant-a", workspace="ws-a")
    pub_b = _publication(tenant="tenant-b", workspace="ws-b")
    id_a = mint_collaborative_activity_id(idempotency_key=pub_a.idempotency_key)
    id_b = mint_collaborative_activity_id(idempotency_key=pub_b.idempotency_key)
    assert id_a != id_b


def test_mp6a_c1_same_canonical_key_same_activity_id() -> None:
    key = _key()
    assert mint_collaborative_activity_id(idempotency_key=key) == mint_collaborative_activity_id(
        idempotency_key=key
    )


def test_mp6a_c1_scope_key_tenant_mismatch_fails() -> None:
    with pytest.raises(ValidationError, match="tenant_id"):
        CollaborativeActivityPublication(
            idempotency_key=_key(tenant="tenant-a"),
            actor=_actor("tenant-a"),
            scope=_scope(tenant="tenant-b"),
            target=WorkItemActivityTargetRef(work_item_id="wi-1"),
            outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
            occurred_at=_NOW,
        )


def test_mp6a_c1_publication_activity_type_is_key_authority() -> None:
    pub = _publication()
    assert pub.activity_type == pub.idempotency_key.activity_type


def test_mp6a_c1_plugin_activity_type_without_core_edit() -> None:
    plugin_type = CollaborativeActivityTypeId.for_extension("acme.corp", "review.approved")
    assert plugin_type.qualified_id == "acme.corp.review.approved"


def test_mp6a_c1_reserved_activity_type_namespace_rejected_for_extension() -> None:
    with pytest.raises(ValueError, match="reserved"):
        CollaborativeActivityTypeId.for_extension("platform", "custom.event")


def test_mp6a_c1_cross_plugin_source_identity_differs() -> None:
    key_a = _key(
        source=CollaborativeActivitySourceId.for_extension("vendor.a", "connector"),
        source_stable_id="same-stable",
    )
    key_b = _key(
        source=CollaborativeActivitySourceId.for_extension("vendor.b", "connector"),
        source_stable_id="same-stable",
    )
    assert mint_collaborative_activity_id(idempotency_key=key_a) != mint_collaborative_activity_id(
        idempotency_key=key_b
    )


def test_mp6a_c1_builtin_platform_types_stable() -> None:
    assert (
        CollaborativeActivityBuiltinType.WORK_ITEM_CREATED.qualified_id
        == "platform.work_item.created"
    )


def test_mp6a_c1_query_accepts_plugin_and_builtin_types() -> None:
    plugin_type = CollaborativeActivityTypeId.for_extension("acme", "signal")
    query = CollaborativeActivityQuery(
        tenant_id="t1",
        workspace_id="w1",
        activity_types=(
            CollaborativeActivityBuiltinType.DECISION_RECORDED,
            plugin_type,
        ),
    )
    assert len(query.activity_types) == 2


def test_mp6a_c1_activity_requires_minted_id_and_aligned_type() -> None:
    key = _key()
    activity_id = mint_collaborative_activity_id(idempotency_key=key)
    activity = CollaborativeActivity(
        activity_id=activity_id,
        idempotency_key=key,
        activity_type=key.activity_type,
        actor=_actor(),
        scope=_scope(),
        target=WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_NOW,
        recorded_at=_NOW,
        append_position=1,
    )
    assert activity.activity_id == activity_id

    with pytest.raises(ValidationError, match="activity_type"):
        CollaborativeActivity(
            activity_id=activity_id,
            idempotency_key=key,
            activity_type=CollaborativeActivityBuiltinType.WORK_ITEM_UPDATED,
            actor=_actor(),
            scope=_scope(),
            target=WorkItemActivityTargetRef(work_item_id="wi-1"),
            outcome=CollaborativeActivityOutcome(
                status=CollaborativeActivityOutcomeStatus.SUCCEEDED
            ),
            occurred_at=_NOW,
            recorded_at=_NOW,
            append_position=1,
        )


def test_mp6a_c1_delegation_fields_bidirectional() -> None:
    with pytest.raises(ValidationError, match="delegation"):
        CollaborativeActivityActorRef(
            tenant_id="tenant-a",
            principal_id="p1",
            principal_kind=PrincipalKind.HUMAN,
            delegator_principal_id="delegator-1",
        )
