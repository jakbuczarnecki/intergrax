# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — final state persistence ordering before worker termination."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_reliability import ExecutionRuntimeShutdownPhase
from testing_support.shutdown.models import ReferenceShutdownFailureKind
from testing_support.shutdown.ports import InMemoryFinalStateStore
from tests.unit.runtime.architecture._ee_b4_b_lifecycle import make_lifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_ee_b4_b_final_state_before_terminate_workers() -> None:
    lifecycle = make_lifecycle()
    outcome = await lifecycle.shutdown()
    assert lifecycle.final_state.persisted
    assert outcome.phase_trail.index(
        ExecutionRuntimeShutdownPhase.PERSIST_FINAL_STATE
    ) < outcome.phase_trail.index(ExecutionRuntimeShutdownPhase.TERMINATE_WORKERS)
    assert outcome.clean_success


@pytest.mark.asyncio
async def test_ee_b4_b_final_state_failure_not_clean() -> None:
    lifecycle = make_lifecycle()
    lifecycle.final_state = InMemoryFinalStateStore(fail_on_call=1)
    outcome = await lifecycle.shutdown()
    assert outcome.primary_failure_kind is ReferenceShutdownFailureKind.FINAL_STATE
    assert not outcome.clean_success
