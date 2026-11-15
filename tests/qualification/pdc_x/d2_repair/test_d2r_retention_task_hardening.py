# © Artur Czarnecki. All rights reserved.
# pyright: reportArgumentType=false, reportCallIssue=false

"""T-IGX-041-B pinned tests: F-D2-13 the retention sweep task bounds its clock and its run time."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from intergrax.background_tasks.registry import TaskRegistry
from intergrax.memory.retention_sweep import (
    MemoryRetentionSweep,
    memory_retention_sweep_task_request,
    register_memory_retention_sweep_task,
)
from tests.qualification.pdc_x.d2.test_d2_memory_retention import (
    _NOW,
    _TENANT,
    _active_ids,
    _harness,
    _identity,
)

_FAR_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)


def _candidate_ids(harness: Any, user_id: str = "user-1") -> set[str]:
    """Every stored entry, expired or not (the sweep's own candidate view)."""
    return {e.entry_id for e in asyncio.run(harness.capability.list_retention_candidates(user_id))}


pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _definition(harness: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
    return register_memory_retention_sweep_task(
        TaskRegistry(),
        sweep=MemoryRetentionSweep(control_plane=harness.plane, candidates=harness.capability),
        identity_resolver=lambda tenant_id, user_id: _identity(user_id),
        clock=lambda: _NOW,
        **kwargs,
    )


def _invoke(definition: Any, *, as_of: Any = None, user_id: str = "user-1") -> Any:
    request = memory_retention_sweep_task_request(tenant_id=_TENANT, run_id="run-1", user_id=user_id, as_of=as_of)
    return definition.handler(tenant_id=_TENANT, run_id="run-1", payload=request.payload)


def test_f_d2_13_a_far_future_as_of_in_the_payload_is_refused_and_nothing_expires() -> None:
    harness = _harness()
    entry = asyncio.run(harness.remember("live", valid_until=_FAR_FUTURE))

    result = _invoke(_definition(harness), as_of=datetime(2100, 1, 1, tzinfo=UTC))

    assert result.success is False
    assert result.error is not None and result.error.error_code == "memory_retention_sweep.as_of_out_of_range"
    assert _candidate_ids(harness) == {entry}
    assert asyncio.run(_active_ids(harness)) == {entry}


def test_f_d2_13_an_as_of_slightly_ahead_of_the_clock_is_accepted() -> None:
    harness = _harness()
    expired = asyncio.run(harness.remember("expired", valid_until=_NOW - timedelta(days=1)))

    result = _invoke(_definition(harness), as_of=_NOW + timedelta(seconds=30))

    assert result.success is True
    assert expired not in asyncio.run(_active_ids(harness))


def test_f_d2_13_a_past_as_of_still_works() -> None:
    harness = _harness()
    expired = asyncio.run(harness.remember("expired", valid_until=_NOW - timedelta(days=5)))
    not_yet = asyncio.run(harness.remember("not-yet", valid_until=_NOW - timedelta(hours=1)))

    result = _invoke(_definition(harness), as_of=_NOW - timedelta(days=2))

    assert result.success is True
    assert _candidate_ids(harness) == {not_yet}
    assert expired not in _candidate_ids(harness)


def test_f_d2_13_without_an_as_of_the_injected_clock_is_used() -> None:
    harness = _harness()
    expired = asyncio.run(harness.remember("expired", valid_until=_NOW - timedelta(days=1)))
    kept = asyncio.run(harness.remember("kept"))

    result = _invoke(_definition(harness))

    assert result.success is True
    assert asyncio.run(_active_ids(harness)) == {kept}
    assert expired not in asyncio.run(_active_ids(harness))


def test_f_d2_13_the_payload_tenant_check_still_rejects_a_forged_tenant() -> None:
    harness = _harness()
    entry = asyncio.run(harness.remember("expired", valid_until=_NOW - timedelta(days=1)))
    definition = _definition(harness)
    forged = json.loads(
        memory_retention_sweep_task_request(tenant_id=_TENANT, run_id="r", user_id="user-1").payload.decode("utf-8")
    )
    forged["tenant_id"] = "tenant-other"

    result = definition.handler(tenant_id=_TENANT, run_id="r", payload=json.dumps(forged).encode("utf-8"))

    assert result.success is False
    assert _candidate_ids(harness) == {entry}


class _NeverFinishingSweep:
    """Sweep stand-in whose run never completes; records cancellation."""

    def __init__(self) -> None:
        self.cancelled = False
        self.started = False

    async def run(self, request: Any) -> Any:
        self.started = True
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def test_f_d2_13_a_sweep_that_exceeds_the_deadline_returns_a_typed_timeout_and_is_cancelled() -> None:
    sweep = _NeverFinishingSweep()
    definition = register_memory_retention_sweep_task(
        TaskRegistry(),
        sweep=sweep,  # type: ignore[arg-type]
        identity_resolver=lambda tenant_id, user_id: _identity(user_id),
        clock=lambda: _NOW,
        deadline_seconds=0.3,
    )

    result = _invoke(definition)

    assert result.success is False
    assert result.error is not None and result.error.error_code == "memory_retention_sweep.timeout"
    assert sweep.started is True
    assert sweep.cancelled is True


def test_f_d2_13_a_sweep_inside_the_deadline_is_not_affected() -> None:
    harness = _harness()
    expired = asyncio.run(harness.remember("expired", valid_until=_NOW - timedelta(days=1)))

    result = _invoke(_definition(harness, deadline_seconds=30.0))

    assert result.success is True
    assert expired not in asyncio.run(_active_ids(harness))
