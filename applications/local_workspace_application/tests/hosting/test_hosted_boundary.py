# © Artur Czarnecki. All rights reserved.

"""APP-HOST-8C — LKW hosting boundary component unit tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import cast

import pytest

from intergrax.hosting import (
    HostedApplicationComponentState,
    HostedApplicationContext,
)
from local_workspace_application.hosting.boundary import (
    LOCAL_WORKSPACE_HOSTING_BOUNDARY_COMPONENT_ID,
    _LocalWorkspaceHostingBoundary,
)

pytestmark = [pytest.mark.unit]


class _FixedClock:
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment

    def advance(self, delta: timedelta) -> None:
        self._moment = self._moment + delta


def _fake_context(
    *,
    application_id: str = "local_workspace",
    clock: _FixedClock | None = None,
) -> HostedApplicationContext:
    resolved_clock = clock or _FixedClock(datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc))
    return cast(
        HostedApplicationContext,
        type(
            "_FakeHostedContext",
            (),
            {
                "application_id": application_id,
                "clock": resolved_clock,
            },
        )(),
    )


@pytest.mark.asyncio
async def test_boundary_lifecycle_and_health_timestamps() -> None:
    clock = _FixedClock(datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc))
    context = _fake_context(clock=clock)
    boundary = _LocalWorkspaceHostingBoundary()

    assert boundary.component_id == LOCAL_WORKSPACE_HOSTING_BOUNDARY_COMPONENT_ID

    initial = await boundary.health(context)
    assert initial.state is HostedApplicationComponentState.CREATED
    assert initial.healthy is False
    assert initial.ready is False
    assert initial.detail_code == "not_started"
    assert initial.last_check_at == clock.now()

    start_at = clock.now()
    await boundary.start(context)
    after_start = await boundary.health(context)
    assert after_start.state is HostedApplicationComponentState.STARTING
    assert after_start.healthy is True
    assert after_start.ready is False
    assert after_start.detail_code == "waiting_before_ready"
    assert after_start.last_transition_at == start_at
    assert after_start.last_check_at == clock.now()

    clock.advance(timedelta(seconds=1))
    hook_at = clock.now()
    await boundary.mark_before_ready(context)
    after_hook = await boundary.health(context)
    assert after_hook.state is HostedApplicationComponentState.READY
    assert after_hook.healthy is True
    assert after_hook.ready is True
    assert after_hook.detail_code == "hosted_boundary_ready"
    assert after_hook.safe_message == "before_ready hook completed"
    assert after_hook.last_transition_at == hook_at
    assert after_hook.last_check_at == clock.now()

    clock.advance(timedelta(seconds=1))
    stop_at = clock.now()
    await boundary.stop(context)
    after_stop = await boundary.health(context)
    assert after_stop.state is HostedApplicationComponentState.STOPPED
    assert after_stop.healthy is True
    assert after_stop.ready is False
    assert after_stop.detail_code == "stopped"
    assert after_stop.last_transition_at == stop_at
    assert after_stop.last_check_at == clock.now()


@pytest.mark.asyncio
async def test_duplicate_active_start_rejected() -> None:
    boundary = _LocalWorkspaceHostingBoundary()
    context = _fake_context()
    await boundary.start(context)
    with pytest.raises(RuntimeError, match="already started"):
        await boundary.start(context)


@pytest.mark.asyncio
async def test_mark_before_ready_before_start_rejected() -> None:
    boundary = _LocalWorkspaceHostingBoundary()
    with pytest.raises(RuntimeError, match="is not started"):
        await boundary.mark_before_ready(_fake_context())


@pytest.mark.asyncio
async def test_repeated_stop_is_harmless() -> None:
    boundary = _LocalWorkspaceHostingBoundary()
    context = _fake_context()
    await boundary.start(context)
    await boundary.stop(context)
    await boundary.stop(context)
    health = await boundary.health(context)
    assert health.state is HostedApplicationComponentState.STOPPED
    assert health.detail_code == "stopped"


@pytest.mark.asyncio
async def test_start_after_stop_resets_boundary() -> None:
    boundary = _LocalWorkspaceHostingBoundary()
    context = _fake_context()
    await boundary.start(context)
    await boundary.mark_before_ready(context)
    await boundary.stop(context)
    await boundary.start(context)
    health = await boundary.health(context)
    assert health.state is HostedApplicationComponentState.STARTING
    assert health.ready is False
    assert health.detail_code == "waiting_before_ready"


@pytest.mark.asyncio
async def test_wrong_application_id_rejected() -> None:
    boundary = _LocalWorkspaceHostingBoundary()
    bad = _fake_context(application_id="other_app")
    with pytest.raises(RuntimeError, match="unexpected application id"):
        await boundary.start(bad)
    with pytest.raises(RuntimeError, match="unexpected application id"):
        await boundary.mark_before_ready(bad)
    with pytest.raises(RuntimeError, match="unexpected application id"):
        await boundary.stop(bad)
