# © Artur Czarnecki. All rights reserved.

"""Execute behavioral eval cases and aggregate summaries."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from tests.qualification.memory_behavior.contracts import (
    BehaviorEvalCase,
    BehaviorGateKind,
    BehaviorViolationLedger,
    MemoryBehaviorEvaluationSummary,
    SemanticQualityMetrics,
)


async def run_behavior_cases(
    cases: Sequence[BehaviorEvalCase],
) -> MemoryBehaviorEvaluationSummary:
    hard_passed = 0
    hard_failed = 0
    for case in cases:
        try:
            await case.runner()
            if case.gate is BehaviorGateKind.HARD:
                hard_passed += 1
        except Exception:
            if case.gate is BehaviorGateKind.HARD:
                hard_failed += 1
            raise
    return MemoryBehaviorEvaluationSummary(
        scenario_count=len(cases),
        hard_passed=hard_passed,
        hard_failed=hard_failed,
        violations=BehaviorViolationLedger().counters,
    )


def run_behavior_cases_sync(cases: Sequence[BehaviorEvalCase]) -> MemoryBehaviorEvaluationSummary:
    return asyncio.run(run_behavior_cases(cases))


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
