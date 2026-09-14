# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — concurrent shutdown single effective lifecycle."""

from __future__ import annotations

import asyncio

import pytest

from tests.unit.runtime.architecture._ee_b4_b_lifecycle import make_lifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_ee_b4_b_concurrent_shutdown_deterministic() -> None:
    lifecycle = make_lifecycle()
    results = await asyncio.gather(lifecycle.shutdown(), lifecycle.shutdown())
    assert results[0] is results[1]
    assert results[0].clean_success
    assert lifecycle.mandatory_evidence.calls == 1
