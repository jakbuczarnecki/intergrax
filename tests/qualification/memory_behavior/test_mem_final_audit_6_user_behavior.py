# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6 — USER scope behavioral gates via MemoryControlPlane."""

from __future__ import annotations

import pytest

from tests.qualification.memory_behavior.contracts import BehaviorEvalContext, BehaviorViolationLedger
from tests.qualification.memory_behavior.scenarios.user import (
    run_user_01_basic_remember_recall,
    run_user_02_irrelevant_memory_not_displacing,
    run_user_03_top_k_respected,
    run_user_04_empty_query_no_semantic_recall,
    run_user_05_disabled_semantic_fallback,
    run_user_06_forget_hard_gate,
    run_user_09_supersession_lineage,
    run_user_10_supersession_recall_prefers_new,
    run_user_11_self_supersession_rejected,
    run_user_12_supersession_missing_target,
    run_user_16_provenance_preserved,
    run_user_18_governance_deny_zero_side_effects,
    run_user_20_conflicting_facts_unresolved,
    run_user_21_supersession_resolves_conflict,
    run_user_22_deterministic_ordering,
)

pytestmark = pytest.mark.gate


def _ctx(ledger: BehaviorViolationLedger | None = None) -> BehaviorEvalContext:
    return BehaviorEvalContext(ledger=ledger or BehaviorViolationLedger())


@pytest.mark.asyncio
async def test_user_01_basic_remember_recall() -> None:
    await run_user_01_basic_remember_recall(_ctx())


@pytest.mark.asyncio
async def test_user_02_irrelevant_memory_not_displacing() -> None:
    await run_user_02_irrelevant_memory_not_displacing(_ctx())


@pytest.mark.asyncio
async def test_user_03_top_k_respected() -> None:
    await run_user_03_top_k_respected(_ctx())


@pytest.mark.asyncio
async def test_user_04_empty_query_no_semantic_recall() -> None:
    await run_user_04_empty_query_no_semantic_recall(_ctx())


@pytest.mark.asyncio
async def test_user_05_disabled_semantic_fallback() -> None:
    await run_user_05_disabled_semantic_fallback(_ctx())


@pytest.mark.asyncio
async def test_user_06_forget_hard_gate(violation_ledger: BehaviorViolationLedger) -> None:
    await run_user_06_forget_hard_gate(_ctx(violation_ledger))


@pytest.mark.asyncio
async def test_user_09_supersession_lineage() -> None:
    await run_user_09_supersession_lineage(_ctx())


@pytest.mark.asyncio
async def test_user_10_supersession_recall_prefers_new(
    violation_ledger: BehaviorViolationLedger,
) -> None:
    await run_user_10_supersession_recall_prefers_new(_ctx(violation_ledger))


@pytest.mark.asyncio
async def test_user_11_self_supersession_rejected() -> None:
    await run_user_11_self_supersession_rejected(_ctx())


@pytest.mark.asyncio
async def test_user_12_supersession_missing_target() -> None:
    await run_user_12_supersession_missing_target(_ctx())


@pytest.mark.asyncio
async def test_user_16_provenance_preserved() -> None:
    await run_user_16_provenance_preserved(_ctx())


@pytest.mark.asyncio
async def test_user_18_governance_deny_zero_side_effects() -> None:
    await run_user_18_governance_deny_zero_side_effects(_ctx())


@pytest.mark.asyncio
async def test_user_20_conflicting_facts_unresolved() -> None:
    await run_user_20_conflicting_facts_unresolved(_ctx())


@pytest.mark.asyncio
async def test_user_21_supersession_resolves_conflict(
    violation_ledger: BehaviorViolationLedger,
) -> None:
    await run_user_21_supersession_resolves_conflict(_ctx(violation_ledger))


@pytest.mark.asyncio
async def test_user_22_deterministic_ordering() -> None:
    await run_user_22_deterministic_ordering(_ctx())
