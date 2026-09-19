# © Artur Czarnecki. All rights reserved.

"""Reusable conformance helpers for ``CollaborativeActivityAppendStore`` implementations."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from intergrax.contracts.collaborative_activity import (
    ActivityIdempotencyKey,
    CollaborativeActivity,
    CollaborativeActivityAppendIntent,
    CollaborativeActivityAppendStore,
    CollaborativeActivityBuiltinSource,
    CollaborativeActivityBuiltinType,
    CollaborativeActivityDurabilityClass,
    CollaborativeActivityPublication,
    CollaborativeActivityScope,
    CollaborativeActivityTargetRef,
    CollaborativeActivityActorRef,
    CollaborativeActivityOutcome,
    CollaborativeActivityOutcomeStatus,
    WorkItemActivityTargetRef,
)
from intergrax.contracts.collaborative_work import PrincipalKind

_FIXED_RECORDED = datetime(2026, 9, 19, 10, 0, 0, tzinfo=timezone.utc)
_OCCURRED = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)


def fixed_recorded_at() -> datetime:
    return _FIXED_RECORDED


def make_publication(
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "ws-a",
    stable_id: str = "stable-1",
    activity_type: CollaborativeActivityBuiltinType = CollaborativeActivityBuiltinType.WORK_ITEM_CREATED,
) -> CollaborativeActivityPublication:
    key = ActivityIdempotencyKey(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source=CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
        source_stable_id=stable_id,
        activity_type=activity_type,
    )
    return CollaborativeActivityPublication(
        idempotency_key=key,
        actor=CollaborativeActivityActorRef(
            tenant_id=tenant_id,
            principal_id="principal-1",
            principal_kind=PrincipalKind.HUMAN,
        ),
        scope=CollaborativeActivityScope(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_item_id="wi-1",
        ),
        target=WorkItemActivityTargetRef(work_item_id="wi-1"),
        outcome=CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED),
        occurred_at=_OCCURRED,
    )


def make_intent(
    publication: CollaborativeActivityPublication | None = None,
    *,
    effective: CollaborativeActivityDurabilityClass = CollaborativeActivityDurabilityClass.COLLABORATIVE,
) -> CollaborativeActivityAppendIntent:
    return CollaborativeActivityAppendIntent(
        publication=publication or make_publication(),
        effective_durability_class=effective,
    )


def run_append_store_contract_suite(
    store_factory: Callable[[], CollaborativeActivityAppendStore],
) -> None:
    store = store_factory()
    intent = make_intent(effective=CollaborativeActivityDurabilityClass.AUDIT_CRITICAL)
    first = store.append_idempotent(intent)
    assert first.append_position == 1
    assert first.recorded_at == _FIXED_RECORDED
    assert first.durability_class == CollaborativeActivityDurabilityClass.AUDIT_CRITICAL

    replay = store.append_idempotent(intent)
    assert replay == first
    assert replay.append_position == first.append_position
    assert replay.recorded_at == first.recorded_at
    assert replay.durability_class == first.durability_class

    changed = make_intent(
        intent.publication.model_copy(
            update={
                "requested_durability_class": CollaborativeActivityDurabilityClass.INFORMATIONAL,
                "occurred_at": _OCCURRED.replace(hour=13),
            }
        ),
        effective=CollaborativeActivityDurabilityClass.INFORMATIONAL,
    )
    assert store.append_idempotent(changed) == first

    second_pub = make_publication(stable_id="stable-2")
    second = store.append_idempotent(make_intent(second_pub))
    assert second.append_position == 2

    other_ws = make_publication(workspace_id="ws-b", stable_id="stable-1")
    other_ws_activity = store.append_idempotent(make_intent(other_ws))
    assert other_ws_activity.append_position == 1

    other_tenant = make_publication(tenant_id="tenant-b", stable_id="stable-1")
    other_tenant_activity = store.append_idempotent(make_intent(other_tenant))
    assert other_tenant_activity.append_position == 1

    loaded = store.get_by_idempotency_key(intent.publication.idempotency_key)
    assert loaded == first


def run_concurrent_duplicate_contract(
    store_factory: Callable[[], CollaborativeActivityAppendStore],
    *,
    open_second_connection: Callable[[], CollaborativeActivityAppendStore] | None = None,
) -> None:
    store_a = store_factory()
    store_b = open_second_connection() if open_second_connection is not None else store_factory()
    intent = make_intent(make_publication(stable_id="race-dup"))
    results: list[CollaborativeActivity] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt(store: CollaborativeActivityAppendStore) -> None:
        try:
            barrier.wait(timeout=5)
            results.append(store.append_idempotent(intent))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=attempt, args=(store_a,)),
        threading.Thread(target=attempt, args=(store_b,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert len(results) == 2
    assert results[0] == results[1]
    assert results[0].append_position == 1


def run_concurrent_distinct_contract(
    store_factory: Callable[[], CollaborativeActivityAppendStore],
    *,
    open_second_connection: Callable[[], CollaborativeActivityAppendStore] | None = None,
    parallel: int = 8,
) -> None:
    store_a = store_factory()
    store_b = open_second_connection() if open_second_connection is not None else store_factory()
    results: list[CollaborativeActivity] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(parallel)

    def attempt(index: int, store: CollaborativeActivityAppendStore) -> None:
        try:
            barrier.wait(timeout=5)
            pub = make_publication(stable_id=f"distinct-{index}")
            results.append(store.append_idempotent(make_intent(pub)))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    stores = [store_a, store_b] * (parallel // 2)
    threads = [
        threading.Thread(target=attempt, args=(index, stores[index % len(stores)]))
        for index in range(parallel)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    positions = {activity.append_position for activity in results}
    assert len(positions) == parallel
    assert positions == set(range(1, parallel + 1))
