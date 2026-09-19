# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6-R2 — behavioral runner aggregation and qualification gates."""

from __future__ import annotations

import pytest

from tests.qualification.memory_behavior.behavior_registry import MEM_AUDIT_6_BEHAVIOR_CASES
from tests.qualification.memory_behavior.contracts import (
    BehaviorEvalCase,
    BehaviorEvalContext,
    BehaviorGateKind,
    BehaviorScenarioCategory,
)
from tests.qualification.memory_behavior.runner import (
    EXPECTED_MEM_AUDIT_6_BEHAVIOR_SCENARIO_COUNT,
    assert_behavior_qualification_pass,
    run_behavior_cases_sync,
    run_mem_final_audit_6_behavioral_qualification_sync,
)
from tests.qualification.memory_behavior.scenarios.user import run_user_01_basic_remember_recall

pytestmark = pytest.mark.gate


async def _record_cross_user(ctx: BehaviorEvalContext) -> None:
    ctx.ledger.record_cross_user_leak()


async def _record_deleted_pair(ctx: BehaviorEvalContext) -> None:
    ctx.ledger.record_deleted_resurrection(2)


async def _record_cross_tenant_leak(ctx: BehaviorEvalContext) -> None:
    ctx.ledger.record_cross_tenant_leak()


def test_behavior_runner_aggregates_real_violation_ledger() -> None:
    cases = (
        BehaviorEvalCase(
            scenario_id="HARNESS-A",
            category=BehaviorScenarioCategory.HARNESS_INTEGRITY,
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
            category=BehaviorScenarioCategory.HARNESS_INTEGRITY,
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
                category=BehaviorScenarioCategory.HARNESS_INTEGRITY,
                gate=BehaviorGateKind.HARD,
                runner=_record_cross_user,
            ),
        )
    )
    assert summary.violations.cross_user_leaks == 1
    with pytest.raises(AssertionError):
        assert_behavior_qualification_pass(summary)


def test_aggregation_mutation_with_real_case_and_synthetic_leak() -> None:
    cases = (
        BehaviorEvalCase(
            scenario_id="USER-01",
            category=BehaviorScenarioCategory.USER,
            gate=BehaviorGateKind.HARD,
            runner=run_user_01_basic_remember_recall,
        ),
        BehaviorEvalCase(
            scenario_id="HARNESS-MUTATION-TENANT",
            category=BehaviorScenarioCategory.HARNESS_INTEGRITY,
            gate=BehaviorGateKind.HARD,
            runner=_record_cross_tenant_leak,
        ),
    )
    ctx = BehaviorEvalContext()
    summary = run_behavior_cases_sync(cases, ctx, fail_fast=False)
    assert summary.violations.cross_tenant_leaks == 1
    with pytest.raises(AssertionError):
        assert_behavior_qualification_pass(summary)


def test_shared_context_ledger_aggregates_across_real_scenarios() -> None:
    ctx = BehaviorEvalContext()
    first = run_behavior_cases_sync(MEM_AUDIT_6_BEHAVIOR_CASES[:2], ctx)
    second = run_behavior_cases_sync(MEM_AUDIT_6_BEHAVIOR_CASES[2:4], ctx)
    assert first.hard_passed == 2
    assert second.hard_passed == 2
    assert not ctx.ledger.counters.has_hard_violation


def test_mem_final_audit_6r2_behavioral_qualification_aggregate_run_1() -> None:
    summary = run_mem_final_audit_6_behavioral_qualification_sync()
    assert summary.scenario_count == EXPECTED_MEM_AUDIT_6_BEHAVIOR_SCENARIO_COUNT
    assert summary.hard_passed == EXPECTED_MEM_AUDIT_6_BEHAVIOR_SCENARIO_COUNT
    assert summary.hard_failed == 0
    assert summary.violations.cross_tenant_leaks == 0
    assert summary.violations.cross_user_leaks == 0
    assert summary.violations.deleted_resurrections == 0
    assert summary.violations.superseded_as_current == 0
    assert summary.violations.projection_only_ghosts == 0
    assert summary.violations.identity_authority_violations == 0
    assert not summary.violations.has_hard_violation
    assert_behavior_qualification_pass(summary)


def test_mem_final_audit_6r2_behavioral_qualification_aggregate_run_2() -> None:
    summary = run_mem_final_audit_6_behavioral_qualification_sync()
    assert summary.scenario_count == EXPECTED_MEM_AUDIT_6_BEHAVIOR_SCENARIO_COUNT
    assert summary.hard_failed == 0
    assert_behavior_qualification_pass(summary)


def test_mem_final_audit_6r2_aggregate_summaries_are_identical() -> None:
    first = run_mem_final_audit_6_behavioral_qualification_sync()
    second = run_mem_final_audit_6_behavioral_qualification_sync()
    assert first.scenario_count == second.scenario_count
    assert first.hard_passed == second.hard_passed
    assert first.hard_failed == second.hard_failed
    assert first.violations == second.violations
