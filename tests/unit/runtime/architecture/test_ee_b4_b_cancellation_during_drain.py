# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — cancellation during drain releases capacity."""

from __future__ import annotations

import asyncio

import pytest

from testing_support.chaos.barriers import PhaseGate
from testing_support.shutdown.models import ReferenceRootAdmissionDecision
from tests.unit.runtime.architecture._ee_b4_b_lifecycle import make_lifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_ee_b4_b_cancellation_during_drain_releases_permit() -> None:
    lifecycle = make_lifecycle(max_roots=1)
    gate = PhaseGate()
    decision, permit = await lifecycle.try_admit_root()
    assert decision is ReferenceRootAdmissionDecision.ADMITTED
    assert permit is not None

    async def blocked() -> None:
        gate.mark_started("held")
        await gate.block()

    handle = await lifecycle.run_root_work(
        execution_id="exec-cancel",
        permit=permit,
        work_factory=blocked,
    )
    await gate.wait_until_started(frozenset({"held"}))
    handle.worker_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle.worker_task
    outcome = await lifecycle.shutdown()
    assert lifecycle.held_root_permits == 0
    assert outcome.clean_success
