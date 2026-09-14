# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — STOP_ACCEPTING_NEW_WORK rejects new root admission."""

from __future__ import annotations

import pytest

from testing_support.shutdown.models import ReferenceRootAdmissionDecision
from tests.unit.runtime.architecture._ee_b4_b_lifecycle import make_lifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_ee_b4_b_stop_accepting_rejects_new_root() -> None:
    lifecycle = make_lifecycle()
    decision, permit = await lifecycle.try_admit_root()
    assert decision is ReferenceRootAdmissionDecision.ADMITTED
    assert permit is not None
    await lifecycle.release_root_permit(permit)
    outcome = await lifecycle.shutdown()
    assert outcome.new_roots_after_stop == 0
    after, after_permit = await lifecycle.try_admit_root()
    assert after is ReferenceRootAdmissionDecision.REJECTED_TERMINATED
    assert after_permit is None
    assert outcome.clean_success
