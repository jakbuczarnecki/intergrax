# © Artur Czarnecki. All rights reserved.

"""Reusable conformance helpers for ``CollaborativeActivityReadPort`` implementations."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from intergrax.collaborative_work.collaborative_activity_page_cursor_codec import (
    encode_collaborative_activity_page_cursor,
)
from intergrax.contracts.collaborative_activity import (
    CollaborativeActivityAppendStore,
    CollaborativeActivityPageCursor,
    CollaborativeActivityQuery,
    CollaborativeActivityReadPort,
)
from intergrax.contracts.collaborative_activity_read import CollaborativeActivityCursorInvalid
from tests.unit.collaborative_work.collaborative_activity_append_store_contract import (
    make_intent,
    make_publication,
)

_LATE_OCCURRED_EARLY = datetime(2026, 9, 18, 10, 0, 0, tzinfo=timezone.utc)
_LATE_OCCURRED_LATE = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)


def _seed_workspace_timeline(
    append_store: CollaborativeActivityAppendStore,
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "ws-a",
    count: int,
    stable_prefix: str = "stable",
) -> None:
    for index in range(count):
        publication = make_publication(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            stable_id=f"{stable_prefix}-{index + 1}",
        )
        append_store.append_idempotent(make_intent(publication))


def _query(
    *,
    tenant_id: str = "tenant-a",
    workspace_id: str = "ws-a",
    limit: int = 2,
    cursor: CollaborativeActivityPageCursor | None = None,
) -> CollaborativeActivityQuery:
    return CollaborativeActivityQuery(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        limit=limit,
        cursor=cursor,
    )


def run_collaborative_activity_read_port_contract_suite(
    *,
    read_port_factory: Callable[[], CollaborativeActivityReadPort],
    append_store_factory: Callable[[], CollaborativeActivityAppendStore],
) -> None:
    append_store = append_store_factory()
    read_port = read_port_factory()
    _seed_workspace_timeline(append_store, count=5)

    page1 = read_port.query(_query(limit=2))
    assert [item.append_position for item in page1.activities] == [1, 2]
    assert page1.next_cursor is not None

    page2 = read_port.query(_query(limit=2, cursor=page1.next_cursor))
    assert [item.append_position for item in page2.activities] == [3, 4]
    assert page2.next_cursor is not None

    page3 = read_port.query(_query(limit=2, cursor=page2.next_cursor))
    assert [item.append_position for item in page3.activities] == [5]
    assert page3.next_cursor is None

    empty = read_port.query(
        _query(
            limit=2,
            cursor=encode_collaborative_activity_page_cursor(
                query=_query(limit=2),
                after_append_position=5,
            ),
        ),
    )
    assert empty.activities == ()
    assert empty.next_cursor is None

    seen: set[int] = set()
    cursor = page1.next_cursor
    while cursor is not None:
        page = read_port.query(_query(limit=2, cursor=cursor))
        for activity in page.activities:
            assert activity.append_position not in seen
            seen.add(activity.append_position)
        cursor = page.next_cursor
    assert seen == {3, 4, 5}


def run_workspace_isolation_read_contract(
    *,
    read_port_factory: Callable[[], CollaborativeActivityReadPort],
    append_store_factory: Callable[[], CollaborativeActivityAppendStore],
) -> None:
    append_store = append_store_factory()
    read_port = read_port_factory()
    _seed_workspace_timeline(append_store, workspace_id="ws-a", count=2, stable_prefix="a")
    _seed_workspace_timeline(append_store, workspace_id="ws-b", count=2, stable_prefix="b")

    page_a = read_port.query(_query(workspace_id="ws-a", limit=10))
    assert all(item.scope.workspace_id == "ws-a" for item in page_a.activities)
    assert len(page_a.activities) == 2

    page_b = read_port.query(_query(workspace_id="ws-b", limit=10))
    assert all(item.scope.workspace_id == "ws-b" for item in page_b.activities)
    assert len(page_b.activities) == 2


def run_tenant_isolation_read_contract(
    *,
    read_port_factory: Callable[[], CollaborativeActivityReadPort],
    append_store_factory: Callable[[], CollaborativeActivityAppendStore],
) -> None:
    append_store = append_store_factory()
    read_port = read_port_factory()
    _seed_workspace_timeline(append_store, tenant_id="tenant-a", count=2)
    _seed_workspace_timeline(
        append_store,
        tenant_id="tenant-b",
        workspace_id="ws-a",
        count=2,
        stable_prefix="tenant-b",
    )

    page_a = read_port.query(_query(tenant_id="tenant-a", limit=10))
    assert all(item.scope.tenant_id == "tenant-a" for item in page_a.activities)

    page_b = read_port.query(_query(tenant_id="tenant-b", workspace_id="ws-a", limit=10))
    assert all(item.scope.tenant_id == "tenant-b" for item in page_b.activities)


def run_late_occurred_at_ordering_contract(
    *,
    read_port_factory: Callable[[], CollaborativeActivityReadPort],
    append_store_factory: Callable[[], CollaborativeActivityAppendStore],
) -> None:
    append_store = append_store_factory()
    read_port = read_port_factory()
    first = append_store.append_idempotent(
        make_intent(
            make_publication(stable_id="late-1").model_copy(
                update={"occurred_at": _LATE_OCCURRED_LATE},
            ),
        ),
    )
    second = append_store.append_idempotent(
        make_intent(
            make_publication(stable_id="late-2").model_copy(
                update={"occurred_at": _LATE_OCCURRED_EARLY},
            ),
        ),
    )
    assert first.append_position == 1
    assert second.append_position == 2

    page = read_port.query(_query(limit=10))
    assert [item.activity_id for item in page.activities] == [
        first.activity_id,
        second.activity_id,
    ]


def run_cursor_scope_mismatch_contract(
    read_port_factory: Callable[[], CollaborativeActivityReadPort],
) -> None:
    read_port = read_port_factory()
    foreign_cursor = encode_collaborative_activity_page_cursor(
        query=_query(tenant_id="tenant-a", workspace_id="ws-other"),
        after_append_position=1,
    )
    with pytest.raises(CollaborativeActivityCursorInvalid):
        read_port.query(_query(workspace_id="ws-a", cursor=foreign_cursor))


def run_invalid_cursor_contract(read_port_factory: Callable[[], CollaborativeActivityReadPort]) -> None:
    read_port = read_port_factory()
    with pytest.raises(CollaborativeActivityCursorInvalid):
        read_port.query(
            _query(
                cursor=CollaborativeActivityPageCursor(token="not-valid-opaque-data"),
            ),
        )
