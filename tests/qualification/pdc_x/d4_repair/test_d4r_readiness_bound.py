# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-D / F-D4-04: the runtime readiness probe is bounded at refresh and at startup."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from intergrax.hosting import (
    HostedApplicationLifecycleState,
    HostedApplicationStartupError,
    LifecyclePolicy,
)
from tests.qualification.pdc_x.d4._support import (
    JournalRuntime,
    build_engine,
    definition_of,
    make_profile,
)

pytestmark = [pytest.mark.integration]

_BOUND_SECONDS = 0.5
_WAIT_SECONDS = 20.0


class _ProbeRuntime(JournalRuntime):
    """Runtime whose ``ready`` can hang or be slow."""

    def __init__(self, *, hang: bool = False, delay: float = 0.0) -> None:
        super().__init__()
        self.hang = hang
        self.delay = delay

    async def ready(self, context) -> bool:  # type: ignore[override]
        if self.hang:
            await asyncio.Event().wait()
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.started


def _profile(runtime: _ProbeRuntime):
    return make_profile(
        runtime=runtime,
        lifecycle=LifecyclePolicy(default_blocking_hook_timeout_seconds=_BOUND_SECONDS),
    )


async def _wait_for(predicate, description: str) -> None:
    deadline = asyncio.get_running_loop().time() + _WAIT_SECONDS
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for: {description}")
        await asyncio.sleep(0.02)


async def test_hung_runtime_ready_after_ready_turns_the_host_not_ready(tmp_path: Path) -> None:
    runtime = _ProbeRuntime()
    engine, _, _ = build_engine(definition_of(_profile(runtime)), tmp_path)
    await engine.start()
    try:
        assert engine.health_snapshot().ready
        runtime.hang = True
        await _wait_for(lambda: engine.health_snapshot().health_evaluation_failed, "health evaluation failure")
        snapshot = engine.health_snapshot()
        assert snapshot.ready is False
        assert snapshot.health_evaluation_failed is True
        reasons = {record.reason_code for record in engine.diagnostics_snapshot().secondary_failures}
        assert "health_poll_failed" in reasons
    finally:
        runtime.hang = False
        await asyncio.wait_for(engine.stop(reason_code="test.cleanup"), _WAIT_SECONDS)


async def test_hung_runtime_ready_at_startup_fails_startup_and_does_not_block_stop(tmp_path: Path) -> None:
    runtime = _ProbeRuntime(hang=True)
    engine, _, _ = build_engine(definition_of(_profile(runtime)), tmp_path)
    with pytest.raises(HostedApplicationStartupError):
        await asyncio.wait_for(engine.start(), _WAIT_SECONDS)
    assert engine.lifecycle_snapshot().state is HostedApplicationLifecycleState.FAILED
    assert engine.diagnostics_snapshot().instance_lease_released
    await asyncio.wait_for(engine.stop(reason_code="test.after_failure"), _WAIT_SECONDS)


async def test_fast_and_slow_but_bounded_ready_stay_ready(tmp_path: Path) -> None:
    """Negative control: a probe within the bound is never failed."""
    runtime = _ProbeRuntime(delay=0.1)
    engine, _, _ = build_engine(definition_of(_profile(runtime)), tmp_path)
    await engine.start()
    try:
        await asyncio.sleep(0.6)
        snapshot = engine.health_snapshot()
        assert snapshot.ready is True
        assert snapshot.health_evaluation_failed is False
    finally:
        await engine.stop(reason_code="test.cleanup")
