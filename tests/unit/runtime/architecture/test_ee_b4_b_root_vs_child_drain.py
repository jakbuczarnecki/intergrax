# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — root vs child admission during shutdown drain."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.contracts.execution_identity import mint_execution_id
from testing_support.chaos.barriers import PhaseGate
from testing_support.shutdown.models import ReferenceRootAdmissionDecision
from tests.unit.runtime.architecture._ee_b4_b_lifecycle import make_lifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_ee_b4_b_child_continuation_allowed_new_root_denied() -> None:
    lifecycle = make_lifecycle()
    gate = PhaseGate()
    parent_id = mint_execution_id()
    decision, permit = await lifecycle.try_admit_root(execution_id=parent_id)
    assert decision is ReferenceRootAdmissionDecision.ADMITTED
    assert permit is not None

    async def parent_work() -> None:
        gate.mark_started("parent")
        await gate.block()

    await lifecycle.run_root_work(
        execution_id=parent_id,
        permit=permit,
        work_factory=parent_work,
    )
    await gate.wait_until_started(frozenset({"parent"}))
    shutdown_task = asyncio.create_task(lifecycle.shutdown())
    await lifecycle.stop_boundary_event.wait()
    child = lifecycle.try_admit_child_continuation(parent_id)
    assert child is ReferenceRootAdmissionDecision.ADMITTED
    new_root, _ = await lifecycle.try_admit_root()
    assert new_root is ReferenceRootAdmissionDecision.REJECTED_SHUTDOWN
    gate.release()
    await shutdown_task
