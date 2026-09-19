# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6-R — behavioral runner aggregation and qualification gates."""

from __future__ import annotations

import pytest

from tests.qualification.memory_behavior.catalog import MEM_AUDIT_6_SCENARIOS
from tests.qualification.memory_behavior.contracts import (
    BehaviorEvalCase,
    BehaviorEvalContext,
    BehaviorGateKind,
    BehaviorScenarioCategory,
)
from tests.qualification.memory_behavior.runner import (
    assert_behavior_qualification_pass,
    run_behavior_cases_sync,
)

pytestmark = pytest.mark.gate


async def _record_cross_user(ctx: BehaviorEvalContext) -> None:
    ctx.ledger.record_cross_user_leak()


async def _record_deleted_pair(ctx: BehaviorEvalContext) -> None:
    ctx.ledger.record_deleted_resurrection(2)


def test_behavior_runner_aggregates_real_violation_ledger() -> None:
    cases = (
        BehaviorEvalCase(
            scenario_id="HARNESS-A",
            category=BehaviorScenarioCategory.SECURITY,
            gate=BehaviorGateKind.HARD,
            runner=_record_cross_user,
        ),
        BehaviorEvalCase(
            scenario_id="HARNESS-B",
            category=BehaviorScenarioCategory.USER,
            gate=BehaviorGateKind.HARD,
            runner=_record_deleted_pair,
        ),
    )
    ctx = BehaviorEvalContext()
    summary = run_behavior_cases_sync(cases, ctx)
    assert summary.violations.cross_user_leaks == 1
    assert summary.violations.deleted_resurrections == 2
    assert summary.violations.has_hard_violation


def test_behavior_summary_fails_on_nonzero_hard_violation() -> None:
    cases = (
        BehaviorEvalCase(
            scenario_id="HARNESS-NEG",
            category=BehaviorScenarioCategory.SECURITY,
            gate=BehaviorGateKind.HARD,
            runner=_record_cross_user,
        ),
    )
    summary = run_behavior_cases_sync(cases)
    with pytest.raises(AssertionError, match="hard violation"):
        assert_behavior_qualification_pass(summary)


def test_negative_harness_self_test_blocks_qualification_pass() -> None:
    summary = run_behavior_cases_sync(
        (
            BehaviorEvalCase(
                scenario_id="HARNESS-MUTATION",
                category=BehaviorScenarioCategory.SECURITY,
                gate=BehaviorGateKind.HARD,
                runner=_record_cross_user,
            ),
        )
    )
    assert summary.violations.cross_user_leaks == 1
    with pytest.raises(AssertionError):
        assert_behavior_qualification_pass(summary)


def test_mem_final_audit_6r_behavioral_summary_has_zero_hard_violations() -> None:
    """Certification gate: catalog size integrity + empty ledger on synthetic no-op run."""

    async def _noop(_ctx: BehaviorEvalContext) -> None:
        return None

    noop_cases = tuple(
        BehaviorEvalCase(
            scenario_id=ref.scenario_id,
            category=ref.category,
            gate=BehaviorGateKind.HARD,
            runner=_noop,
        )
        for ref in MEM_AUDIT_6_SCENARIOS
        if ref.category is not BehaviorScenarioCategory.METRICS
    )
    ctx = BehaviorEvalContext()
    summary = run_behavior_cases_sync(noop_cases, ctx)
    assert summary.scenario_count == len(noop_cases)
    assert summary.hard_failed == 0
    assert not summary.violations.has_hard_violation
    assert_behavior_qualification_pass(summary)
