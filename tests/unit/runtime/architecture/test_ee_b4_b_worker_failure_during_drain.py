# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — worker failure during shutdown drain is contained."""

from __future__ import annotations

import pytest

from testing_support.chaos.barriers import PhaseGate
from testing_support.shutdown.models import (
    ReferenceRootAdmissionDecision,
    ReferenceShutdownFailureKind,
)
from tests.unit.runtime.architecture._ee_b4_b_lifecycle import make_lifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_ee_b4_b_worker_failure_during_drain_no_deadlock() -> None:
    lifecycle = make_lifecycle(drain_timeout_seconds=5.0)
    gate = PhaseGate()
    decision, permit = await lifecycle.try_admit_root()
    assert decision is ReferenceRootAdmissionDecision.ADMITTED
    assert permit is not None

    async def failing_work() -> None:
        gate.mark_started("fail")
        raise RuntimeError("worker_primary_fault")

    handle = await lifecycle.run_root_work(
        execution_id="exec-fail",
        permit=permit,
        work_factory=failing_work,
    )
    await gate.wait_until_started(frozenset({"fail"}))
    outcome = await lifecycle.shutdown()
    with pytest.raises(RuntimeError, match="worker_primary_fault"):
        await handle.worker_task
    assert outcome.primary_failure_kind is ReferenceShutdownFailureKind.WORKER
    assert not outcome.clean_success
    assert lifecycle.managed_worker_count == 0
