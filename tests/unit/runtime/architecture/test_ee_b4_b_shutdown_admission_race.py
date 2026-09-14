# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — shutdown vs new root admission race (event-synchronized)."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.contracts.execution_identity import mint_execution_id
from testing_support.chaos.barriers import PhaseGate
from testing_support.shutdown.models import ReferenceRootAdmissionDecision
from tests.unit.runtime.architecture._ee_b4_b_lifecycle import make_lifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_ee_b4_b_no_root_starts_after_stop_boundary() -> None:
    lifecycle = make_lifecycle()
    gate = PhaseGate()
    decision, permit = await lifecycle.try_admit_root()
    assert decision is ReferenceRootAdmissionDecision.ADMITTED
    assert permit is not None
    rejected = asyncio.Event()

    async def hold_drain() -> None:
        gate.mark_started("active")
        await gate.block()

    await lifecycle.run_root_work(
        execution_id=mint_execution_id(),
        permit=permit,
        work_factory=hold_drain,
    )
    await gate.wait_until_started(frozenset({"active"}))

    async def attempt_admit_after_stop_boundary() -> None:
        await lifecycle.stop_boundary_event.wait()
        decision, _ = await lifecycle.try_admit_root()
        if decision is ReferenceRootAdmissionDecision.REJECTED_SHUTDOWN:
            rejected.set()

    racer = asyncio.create_task(attempt_admit_after_stop_boundary())
    shutdown_task = asyncio.create_task(lifecycle.shutdown())
    await lifecycle.stop_boundary_event.wait()
    await asyncio.wait_for(racer, timeout=5.0)
    assert rejected.is_set()
    gate.release()
    outcome = await shutdown_task
    assert outcome.new_roots_after_stop >= 1
