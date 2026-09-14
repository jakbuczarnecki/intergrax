# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — root capacity permit release on drain paths."""

from __future__ import annotations

import pytest

from testing_support.shutdown.models import ReferenceRootAdmissionDecision
from tests.unit.runtime.architecture._ee_b4_b_lifecycle import make_lifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_ee_b4_b_capacity_released_after_shutdown() -> None:
    lifecycle = make_lifecycle(max_roots=1)
    decision, permit = await lifecycle.try_admit_root()
    assert decision is ReferenceRootAdmissionDecision.ADMITTED
    assert permit is not None

    async def noop() -> None:
        return None

    handle = await lifecycle.run_root_work(
        execution_id="exec-cap",
        permit=permit,
        work_factory=noop,
    )
    await handle.worker_task
    outcome = await lifecycle.shutdown()
    assert lifecycle.held_root_permits == 0
    assert outcome.clean_success
    second, second_permit = await lifecycle.try_admit_root()
    assert second is ReferenceRootAdmissionDecision.REJECTED_TERMINATED


@pytest.mark.asyncio
async def test_ee_b4_b_double_release_counted_idempotent() -> None:
    lifecycle = make_lifecycle()
    decision, permit = await lifecycle.try_admit_root()
    assert permit is not None
    await lifecycle.release_root_permit(permit)
    await lifecycle.release_root_permit(permit)
    outcome = await lifecycle.shutdown()
    assert outcome.double_release_attempts == 1
    assert lifecycle.held_root_permits == 0
