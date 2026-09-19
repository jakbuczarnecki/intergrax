# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6 — identity and isolation hard gates."""

from __future__ import annotations

import pytest

from tests.qualification.memory_behavior.contracts import BehaviorEvalContext, BehaviorViolationLedger
from tests.qualification.memory_behavior.scenarios.security import (
    run_sec_01_identity_user_spoof_denied,
    run_sec_02_identity_tenant_spoof_denied,
    run_sec_03_cross_tenant_shared_backing_reverse_direction,
    run_user_23_cross_user_isolation,
    run_user_24_cross_tenant_isolation,
)

pytestmark = pytest.mark.gate


def _ctx(ledger: BehaviorViolationLedger) -> BehaviorEvalContext:
    return BehaviorEvalContext(ledger=ledger)


@pytest.mark.asyncio
async def test_identity_user_spoof_denied(violation_ledger: BehaviorViolationLedger) -> None:
    await run_sec_01_identity_user_spoof_denied(_ctx(violation_ledger))


@pytest.mark.asyncio
async def test_identity_tenant_spoof_denied(violation_ledger: BehaviorViolationLedger) -> None:
    await run_sec_02_identity_tenant_spoof_denied(_ctx(violation_ledger))


@pytest.mark.asyncio
async def test_cross_user_isolation(violation_ledger: BehaviorViolationLedger) -> None:
    await run_user_23_cross_user_isolation(_ctx(violation_ledger))


@pytest.mark.asyncio
async def test_cross_tenant_isolation(violation_ledger: BehaviorViolationLedger) -> None:
    await run_user_24_cross_tenant_isolation(_ctx(violation_ledger))


@pytest.mark.asyncio
async def test_cross_tenant_shared_backing_reverse_direction(
    violation_ledger: BehaviorViolationLedger,
) -> None:
    await run_sec_03_cross_tenant_shared_backing_reverse_direction(_ctx(violation_ledger))
