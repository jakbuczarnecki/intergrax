# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6 — SESSION episodic path via MemoryControlPlane."""

from __future__ import annotations

import pytest

from tests.qualification.memory_behavior.contracts import BehaviorEvalContext
from tests.qualification.memory_behavior.scenarios.session import (
    run_session_01_basic_episodic_recall,
    run_session_02_session_isolation,
    run_session_03_cross_session_when_enabled,
    run_session_05_scope_authority_on_recall,
)

pytestmark = pytest.mark.gate


@pytest.mark.asyncio
async def test_session_01_basic_episodic_recall() -> None:
    await run_session_01_basic_episodic_recall(BehaviorEvalContext())


@pytest.mark.asyncio
async def test_session_02_session_isolation() -> None:
    await run_session_02_session_isolation(BehaviorEvalContext())


@pytest.mark.asyncio
async def test_session_03_cross_session_when_enabled() -> None:
    await run_session_03_cross_session_when_enabled(BehaviorEvalContext())


@pytest.mark.asyncio
async def test_session_05_scope_authority_on_recall() -> None:
    await run_session_05_scope_authority_on_recall(BehaviorEvalContext())
