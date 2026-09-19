# © Artur Czarnecki. All rights reserved.

"""Execute behavioral eval cases and aggregate summaries."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from tests.qualification.memory_behavior.behavior_registry import MEM_AUDIT_6_BEHAVIOR_CASES
from tests.qualification.memory_behavior.contracts import (
    BehaviorEvalCase,
    BehaviorEvalContext,
    BehaviorGateKind,
    MemoryBehaviorEvaluationSummary,
    SemanticQualityMetrics,
)

EXPECTED_MEM_AUDIT_6_BEHAVIOR_SCENARIO_COUNT = len(MEM_AUDIT_6_BEHAVIOR_CASES)


def assert_behavior_qualification_pass(summary: MemoryBehaviorEvaluationSummary) -> None:
    if summary.hard_failed:
        raise AssertionError(f"hard scenarios failed: {summary.hard_failed}")
    if summary.violations.has_hard_violation:
        raise AssertionError(f"hard violation counters: {summary.violations}")


async def run_behavior_cases(
    cases: Sequence[BehaviorEvalCase],
    ctx: BehaviorEvalContext | None = None,
    *,
    fail_fast: bool = True,
) -> MemoryBehaviorEvaluationSummary:
    eval_ctx = ctx or BehaviorEvalContext()
    hard_passed = 0
    hard_failed = 0
    for case in cases:
        try:
            await case.runner(eval_ctx)
            if case.gate is BehaviorGateKind.HARD:
                hard_passed += 1
        except Exception:
            if case.gate is BehaviorGateKind.HARD:
                hard_failed += 1
            if fail_fast:
                raise
    return MemoryBehaviorEvaluationSummary(
        scenario_count=len(cases),
        hard_passed=hard_passed,
        hard_failed=hard_failed,
        violations=eval_ctx.ledger.counters,
    )


def run_behavior_cases_sync(
    cases: Sequence[BehaviorEvalCase],
    ctx: BehaviorEvalContext | None = None,
    *,
    fail_fast: bool = True,
) -> MemoryBehaviorEvaluationSummary:
    return asyncio.run(run_behavior_cases(cases, ctx, fail_fast=fail_fast))


async def run_mem_final_audit_6_behavioral_qualification() -> MemoryBehaviorEvaluationSummary:
    """Execute all Memory behavioral scenarios on one shared evidence context."""
    ctx = BehaviorEvalContext()
    return await run_behavior_cases(MEM_AUDIT_6_BEHAVIOR_CASES, ctx, fail_fast=True)


def run_mem_final_audit_6_behavioral_qualification_sync() -> MemoryBehaviorEvaluationSummary:
    return asyncio.run(run_mem_final_audit_6_behavioral_qualification())


def build_semantic_metrics(
    *,
    expected_ids: Sequence[str],
    retrieved_ids: Sequence[Sequence[str]],
    top_k: int,
) -> SemanticQualityMetrics:
    hits = 0
    reciprocal_sum = 0.0
    for index, expected in enumerate(expected_ids):
        ranked = retrieved_ids[index] if index < len(retrieved_ids) else ()
        if ranked and ranked[0] == expected:
            hits += 1
        for rank, entry_id in enumerate(ranked[:top_k], start=1):
            if entry_id == expected:
                reciprocal_sum += 1.0 / rank
                break
        in_top_k = expected in ranked[:top_k]
        if not in_top_k and expected:
            pass
    n = max(1, len(expected_ids))
    recall_hits = sum(
        1
        for index, expected in enumerate(expected_ids)
        if expected in (retrieved_ids[index][:top_k] if index < len(retrieved_ids) else ())
    )
    return SemanticQualityMetrics(
        dataset_size=len(expected_ids),
        hit_at_1=hits / n,
        recall_at_k=recall_hits / n,
        mrr=reciprocal_sum / n,
        top_k=top_k,
    )
