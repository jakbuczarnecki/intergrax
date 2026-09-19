# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6 — projection lifecycle and canonical authority gates."""

from __future__ import annotations

import pytest

from tests.qualification.memory_behavior.contracts import BehaviorEvalContext, BehaviorViolationLedger
from tests.qualification.memory_behavior.scenarios.projection import (
    run_p01_partial_projection_remember,
    run_p02_partial_projection_forget,
    run_p03_reconciliation_repair,
    run_p04_reconciliation_idempotent,
    run_p05_reconciliation_projection_failure,
    run_user_08_stale_vector_after_delete,
)

pytestmark = pytest.mark.gate


def _ctx(ledger: BehaviorViolationLedger | None = None) -> BehaviorEvalContext:
    return BehaviorEvalContext(ledger=ledger or BehaviorViolationLedger())


@pytest.mark.asyncio
async def test_p01_partial_projection_remember() -> None:
    await run_p01_partial_projection_remember(_ctx())


@pytest.mark.asyncio
async def test_p02_partial_projection_forget() -> None:
    await run_p02_partial_projection_forget(_ctx())


@pytest.mark.asyncio
async def test_p03_reconciliation_repair() -> None:
    await run_p03_reconciliation_repair(_ctx())


@pytest.mark.asyncio
async def test_p04_reconciliation_idempotent() -> None:
    await run_p04_reconciliation_idempotent(_ctx())


@pytest.mark.asyncio
async def test_user_08_stale_vector_after_delete(violation_ledger: BehaviorViolationLedger) -> None:
    await run_user_08_stale_vector_after_delete(_ctx(violation_ledger))


@pytest.mark.asyncio
async def test_p05_reconciliation_projection_failure() -> None:
    await run_p05_reconciliation_projection_failure(_ctx())
