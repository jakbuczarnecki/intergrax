# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — idempotent shutdown invocations."""

from __future__ import annotations

import pytest

from tests.unit.runtime.architecture._ee_b4_b_lifecycle import make_lifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_ee_b4_b_sequential_shutdown_idempotent() -> None:
    lifecycle = make_lifecycle()
    first = await lifecycle.shutdown()
    second = await lifecycle.shutdown()
    assert first is second
    assert lifecycle.mandatory_evidence.calls == 1
    assert lifecycle.final_state.calls == 1
