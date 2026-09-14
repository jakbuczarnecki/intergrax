# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — managed worker/task leak checks after shutdown."""

from __future__ import annotations

import pytest

from testing_support.shutdown.models import ReferenceRootAdmissionDecision
from tests.unit.runtime.architecture._ee_b4_b_lifecycle import make_lifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_ee_b4_b_no_managed_worker_or_task_leaks() -> None:
    lifecycle = make_lifecycle()
    for index in range(2):
        decision, permit = await lifecycle.try_admit_root()
        assert decision is ReferenceRootAdmissionDecision.ADMITTED
        assert permit is not None

        async def noop() -> None:
            return None

        handle = await lifecycle.run_root_work(
            execution_id=f"exec-{index}",
            permit=permit,
            work_factory=noop,
        )
        await handle.worker_task
    outcome = await lifecycle.shutdown()
    assert outcome.managed_worker_count == 0
    assert outcome.managed_task_count == 0
