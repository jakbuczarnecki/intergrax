# © Artur Czarnecki. All rights reserved.

"""MP-6F — Collaborative Activity source adapter and mapping tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from intergrax.collaborative_work.collaborative_activity_source_adapters import (
    CollaborativeActivityPublicationFailurePolicy,
    CollaborativeActivitySourcePublicationSideEffect,
    CollaborativeWorkServiceWithActivityPublication,
)
from intergrax.collaborative_work.collaborative_activity_source_mapping import (
    CollaborativeActivitySourceMappingError,
    FixedCollaborativeActivityActorPrincipalKindResolver,
    map_work_item_created_publication,
)
from intergrax.collaborative_work.repository import WorkItemNotFound
from intergrax.contracts.collaborative_activity import (
    ActivityIdempotencyKey,
    CollaborativeActivity,
    CollaborativeActivityBuiltinSource,
    CollaborativeActivityBuiltinType,
    CollaborativeActivityPublication,
    mint_collaborative_activity_id,
)
from intergrax.contracts.collaborative_work import (
    CreateWorkItemRequest,
    PrincipalKind,
    WorkItem,
    WorkItemState,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_ACTING = "semantic-actor-principal"
_NOW = datetime(2026, 9, 19, 10, 0, tzinfo=UTC)


@dataclass
class _PublicationSpy:
    calls: list[CollaborativeActivityPublication] = field(default_factory=list)
    fail_next: bool = False

    def publish(self, publication: CollaborativeActivityPublication) -> CollaborativeActivity:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("publication failed")
        self.calls.append(publication)
        activity_id = mint_collaborative_activity_id(idempotency_key=publication.idempotency_key)
        return CollaborativeActivity(
            activity_id=activity_id,
            idempotency_key=publication.idempotency_key,
            activity_type=publication.activity_type,
            actor=publication.actor,
            scope=publication.scope,
            target=publication.target,
            outcome=publication.outcome,
            occurred_at=publication.occurred_at,
            recorded_at=_NOW,
            append_position=1,
            durability_class=publication.requested_durability_class,
        )


class _StubCollaborativeWorkService:
    def __init__(self, *, work_item: WorkItem | None = None, fail: bool = False) -> None:
        self._work_item = work_item
        self._fail = fail
        self.create_calls = 0

    def create_work_item(self, request: CreateWorkItemRequest) -> WorkItem:
        self.create_calls += 1
        if self._fail:
            raise WorkItemNotFound("boom")
        assert self._work_item is not None
        return self._work_item

    def transition_work_item(self, request: object) -> WorkItem:
        raise NotImplementedError

    def create_assignment(self, request: object) -> object:
        raise NotImplementedError

    def transition_assignment(self, request: object) -> object:
        raise NotImplementedError


def _work_item(*, work_item_id: str = "wi-1") -> WorkItem:
    return WorkItem(
        work_item_id=work_item_id,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        created_by_principal_id=_ACTING,
        state=WorkItemState.OPEN,
        revision=0,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _create_request(*, idempotency_key: str = "op-create-1") -> CreateWorkItemRequest:
    return CreateWorkItemRequest(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id="wi-1",
        acting_principal_id=_ACTING,
        idempotency_key=idempotency_key,
    )


def _principal_kind_resolver() -> FixedCollaborativeActivityActorPrincipalKindResolver:
    return FixedCollaborativeActivityActorPrincipalKindResolver(principal_kind=PrincipalKind.HUMAN)


def test_mp6f_work_item_created_mapping_fields() -> None:
    request = _create_request()
    work_item = _work_item()
    publication = map_work_item_created_publication(
        request=request,
        work_item=work_item,
        principal_kind_resolver=_principal_kind_resolver(),
    )
    assert publication.activity_type == CollaborativeActivityBuiltinType.WORK_ITEM_CREATED
    assert publication.idempotency_key.source == CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK
    assert publication.idempotency_key.source_stable_id == "op-create-1"
    assert publication.actor.principal_id == _ACTING
    assert publication.actor.principal_kind is PrincipalKind.HUMAN
    assert publication.scope.tenant_id == _TENANT
    assert publication.scope.work_item_id == "wi-1"
    assert publication.occurred_at == work_item.created_at
    assert publication.target.kind == "work_item"


def test_mp6f_idempotency_key_deterministic_for_same_source_operation() -> None:
    request = _create_request(idempotency_key="replay-key")
    work_item = _work_item()
    resolver = _principal_kind_resolver()
    first = map_work_item_created_publication(
        request=request,
        work_item=work_item,
        principal_kind_resolver=resolver,
    )
    second = map_work_item_created_publication(
        request=request,
        work_item=work_item,
        principal_kind_resolver=resolver,
    )
    assert first.idempotency_key == second.idempotency_key


def test_mp6f_different_idempotency_keys_do_not_collide() -> None:
    work_item = _work_item()
    resolver = _principal_kind_resolver()
    pub_a = map_work_item_created_publication(
        request=_create_request(idempotency_key="op-a"),
        work_item=work_item,
        principal_kind_resolver=resolver,
    )
    pub_b = map_work_item_created_publication(
        request=_create_request(idempotency_key="op-b"),
        work_item=work_item,
        principal_kind_resolver=resolver,
    )
    assert pub_a.idempotency_key != pub_b.idempotency_key


def test_mp6f_scope_mismatch_raises_mapping_error() -> None:
    request = _create_request()
    work_item = _work_item()
    mismatched = work_item.model_copy(update={"tenant_id": "other-tenant"})
    with pytest.raises(CollaborativeActivitySourceMappingError):
        map_work_item_created_publication(
            request=request,
            work_item=mismatched,
            principal_kind_resolver=_principal_kind_resolver(),
        )


def test_mp6f_actor_is_semantic_actor_not_publication_port_identity() -> None:
    """Publisher identity is MP-6C; actor must come from source request truth."""
    request = _create_request()
    work_item = _work_item()
    publication = map_work_item_created_publication(
        request=request,
        work_item=work_item,
        principal_kind_resolver=_principal_kind_resolver(),
    )
    assert publication.actor.principal_id == request.acting_principal_id
    assert publication.actor.principal_id != "mp6c-publisher-principal"


class _CustomMapper:
    def map_work_item_created(
        self,
        *,
        request: CreateWorkItemRequest,
        work_item: WorkItem,
    ) -> CollaborativeActivityPublication:
        base = map_work_item_created_publication(
            request=request,
            work_item=work_item,
            principal_kind_resolver=_principal_kind_resolver(),
        )
        key = ActivityIdempotencyKey(
            tenant_id=base.idempotency_key.tenant_id,
            workspace_id=base.idempotency_key.workspace_id,
            source=base.idempotency_key.source,
            source_stable_id=f"custom:{base.idempotency_key.source_stable_id}",
            activity_type=base.idempotency_key.activity_type,
        )
        return base.model_copy(update={"idempotency_key": key})


def test_mp6f_replaceable_mapper_used_by_adapter() -> None:
    spy = _PublicationSpy()
    inner = _StubCollaborativeWorkService(work_item=_work_item())
    side_effect = CollaborativeActivitySourcePublicationSideEffect(publication_port=spy)
    wrapped = CollaborativeWorkServiceWithActivityPublication(
        inner=inner,  # type: ignore[arg-type]
        side_effect=side_effect,
        mapper=_CustomMapper(),  # type: ignore[arg-type]
    )
    result = wrapped.create_work_item(_create_request(idempotency_key="idem-1"))
    assert result.work_item_id == "wi-1"
    assert len(spy.calls) == 1
    assert spy.calls[0].idempotency_key.source_stable_id == "custom:idem-1"


def test_mp6f_successful_source_operation_emits_one_publication() -> None:
    spy = _PublicationSpy()
    inner = _StubCollaborativeWorkService(work_item=_work_item())
    from intergrax.collaborative_work.collaborative_activity_source_mapping import (
        DefaultCollaborativeWorkActivitySourceMapper,
    )

    side_effect = CollaborativeActivitySourcePublicationSideEffect(publication_port=spy)
    wrapped = CollaborativeWorkServiceWithActivityPublication(
        inner=inner,  # type: ignore[arg-type]
        side_effect=side_effect,
        mapper=DefaultCollaborativeWorkActivitySourceMapper(
            principal_kind_resolver=_principal_kind_resolver(),
        ),
    )
    wrapped.create_work_item(_create_request())
    assert len(spy.calls) == 1


def test_mp6f_failed_source_operation_emits_zero_publications() -> None:
    spy = _PublicationSpy()
    inner = _StubCollaborativeWorkService(fail=True)
    from intergrax.collaborative_work.collaborative_activity_source_mapping import (
        DefaultCollaborativeWorkActivitySourceMapper,
    )

    side_effect = CollaborativeActivitySourcePublicationSideEffect(publication_port=spy)
    wrapped = CollaborativeWorkServiceWithActivityPublication(
        inner=inner,  # type: ignore[arg-type]
        side_effect=side_effect,
        mapper=DefaultCollaborativeWorkActivitySourceMapper(
            principal_kind_resolver=_principal_kind_resolver(),
        ),
    )
    with pytest.raises(WorkItemNotFound):
        wrapped.create_work_item(_create_request())
    assert spy.calls == []


def test_mp6f_publication_failure_raise_policy_propagates() -> None:
    spy = _PublicationSpy(fail_next=True)
    inner = _StubCollaborativeWorkService(work_item=_work_item())
    from intergrax.collaborative_work.collaborative_activity_source_mapping import (
        DefaultCollaborativeWorkActivitySourceMapper,
    )

    side_effect = CollaborativeActivitySourcePublicationSideEffect(
        publication_port=spy,
        failure_policy=CollaborativeActivityPublicationFailurePolicy.RAISE,
    )
    wrapped = CollaborativeWorkServiceWithActivityPublication(
        inner=inner,  # type: ignore[arg-type]
        side_effect=side_effect,
        mapper=DefaultCollaborativeWorkActivitySourceMapper(
            principal_kind_resolver=_principal_kind_resolver(),
        ),
    )
    with pytest.raises(RuntimeError, match="publication failed"):
        wrapped.create_work_item(_create_request())
    assert inner.create_calls == 1


def test_mp6f_publication_failure_log_and_continue_preserves_source_result() -> None:
    spy = _PublicationSpy(fail_next=True)
    inner = _StubCollaborativeWorkService(work_item=_work_item())
    from intergrax.collaborative_work.collaborative_activity_source_mapping import (
        DefaultCollaborativeWorkActivitySourceMapper,
    )

    side_effect = CollaborativeActivitySourcePublicationSideEffect(
        publication_port=spy,
        failure_policy=CollaborativeActivityPublicationFailurePolicy.LOG_AND_CONTINUE,
    )
    wrapped = CollaborativeWorkServiceWithActivityPublication(
        inner=inner,  # type: ignore[arg-type]
        side_effect=side_effect,
        mapper=DefaultCollaborativeWorkActivitySourceMapper(
            principal_kind_resolver=_principal_kind_resolver(),
        ),
    )
    result = wrapped.create_work_item(_create_request())
    assert result.work_item_id == "wi-1"


def test_mp6f_replaceable_publication_port_spy() -> None:
    class _AltPort:
        def __init__(self) -> None:
            self.seen: list[CollaborativeActivityPublication] = []

        def publish(self, publication: CollaborativeActivityPublication) -> CollaborativeActivity:
            self.seen.append(publication)
            return _PublicationSpy().publish(publication)

    alt = _AltPort()
    inner = _StubCollaborativeWorkService(work_item=_work_item())
    from intergrax.collaborative_work.collaborative_activity_source_mapping import (
        DefaultCollaborativeWorkActivitySourceMapper,
    )

    wrapped = CollaborativeWorkServiceWithActivityPublication(
        inner=inner,  # type: ignore[arg-type]
        side_effect=CollaborativeActivitySourcePublicationSideEffect(publication_port=alt),
        mapper=DefaultCollaborativeWorkActivitySourceMapper(
            principal_kind_resolver=_principal_kind_resolver(),
        ),
    )
    wrapped.create_work_item(_create_request())
    assert len(alt.seen) == 1
