# © Artur Czarnecki. All rights reserved.

"""SELF-HEALING R5.2 strategy quality evaluation foundation."""

from __future__ import annotations

import importlib
import inspect
from datetime import datetime, timezone

import pytest

from intergrax.contracts.self_healing.performance_memory import (
    SelfHealingStrategyExecutionOutcome,
    SelfHealingStrategyPerformanceExperience,
    StrategyPerformanceMemoryRepository,
    mint_self_healing_strategy_performance_experience_id,
)
from intergrax.contracts.self_healing.quality_evaluation import (
    StrategyQualityAssessment,
    StrategyQualityEvaluationCriteria,
    StrategyQualityEvaluator,
    summarize_strategy_performance_experiences,
)
from intergrax.runtime.self_healing.performance_memory import InMemoryStrategyPerformanceMemoryRepository
from intergrax.runtime.self_healing.quality_evaluation import (
    BasicStrategyQualityEvaluator,
    StrategyQualityEvaluationService,
    WeightedStrategyQualityEvaluator,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_STRATEGY_ID = "platform.database.reconnect"
_TENANT_ID = "tenant-a"


def _experience(
    outcome: SelfHealingStrategyExecutionOutcome,
    recovery_seconds: float,
    *,
    evidence_suffix: str = "1",
) -> SelfHealingStrategyPerformanceExperience:
    return SelfHealingStrategyPerformanceExperience(
        experience_id=mint_self_healing_strategy_performance_experience_id(),
        tenant_id=_TENANT_ID,
        strategy_id=_STRATEGY_ID,
        workflow_id="sh_wf_eval00000001",
        plan_id="sh_plan_eval000001",
        execution_ids=("exec-1",),
        diagnostic_investigation_id="inv-1",
        execution_outcome=outcome,
        rollback_executed=outcome is SelfHealingStrategyExecutionOutcome.ROLLED_BACK,
        recovery_time_seconds=recovery_seconds,
        evidence_refs=(f"evidence://run/{evidence_suffix}",),
        recorded_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )


def _criteria() -> StrategyQualityEvaluationCriteria:
    return StrategyQualityEvaluationCriteria(tenant_id=_TENANT_ID, strategy_id=_STRATEGY_ID)


def test_evaluator_protocol_is_runtime_checkable() -> None:
    basic = BasicStrategyQualityEvaluator()
    weighted = WeightedStrategyQualityEvaluator()
    assert isinstance(basic, StrategyQualityEvaluator)
    assert isinstance(weighted, StrategyQualityEvaluator)


def test_empty_history_returns_zeroed_assessment() -> None:
    service = StrategyQualityEvaluationService(
        repository=InMemoryStrategyPerformanceMemoryRepository(),
        evaluator=BasicStrategyQualityEvaluator(),
    )
    assessment = service.assess(_criteria())
    assert assessment.execution_count == 0
    assert assessment.successful_executions == 0
    assert assessment.failed_executions == 0
    assert assessment.success_ratio == 0.0
    assert assessment.quality_score == 0.0
    assert assessment.average_recovery_time_seconds is None
    assert assessment.evidence_refs == ()


def test_basic_evaluator_success_ratio_and_recovery_average() -> None:
    experiences = (
        _experience(SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED, 6.0, evidence_suffix="a"),
        _experience(SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED, 10.0, evidence_suffix="b"),
        _experience(SelfHealingStrategyExecutionOutcome.REPAIR_FAILED, 8.0, evidence_suffix="c"),
    )
    repo = InMemoryStrategyPerformanceMemoryRepository()
    for row in experiences:
        repo.append(row)
    service = StrategyQualityEvaluationService(
        repository=repo,
        evaluator=BasicStrategyQualityEvaluator(),
    )
    assessment = service.assess(_criteria())
    assert assessment.execution_count == 3
    assert assessment.successful_executions == 2
    assert assessment.failed_executions == 1
    assert assessment.success_ratio == pytest.approx(2 / 3)
    assert assessment.quality_score == pytest.approx(2 / 3)
    assert assessment.average_recovery_time_seconds == pytest.approx(8.0)
    assert assessment.min_recovery_time_seconds == 6.0
    assert assessment.max_recovery_time_seconds == 10.0
    assert assessment.evidence_refs == (
        "evidence://run/a",
        "evidence://run/b",
        "evidence://run/c",
    )


def test_mixed_outcomes_count_all_terminal_classes() -> None:
    experiences = (
        _experience(SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED, 1.0),
        _experience(SelfHealingStrategyExecutionOutcome.REPAIR_FAILED, 2.0),
        _experience(SelfHealingStrategyExecutionOutcome.ROLLED_BACK, 3.0),
        _experience(SelfHealingStrategyExecutionOutcome.INCONCLUSIVE, 4.0),
    )
    stats = summarize_strategy_performance_experiences(experiences)
    assert stats.successful_executions == 1
    assert stats.failed_executions == 1
    assert stats.rolled_back_executions == 1
    assert stats.inconclusive_executions == 1
    assert stats.success_ratio == 0.25


def test_evaluation_is_deterministic() -> None:
    experiences = (
        _experience(SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED, 5.0),
        _experience(SelfHealingStrategyExecutionOutcome.REPAIR_FAILED, 15.0),
    )
    evaluator = BasicStrategyQualityEvaluator()
    criteria = _criteria()
    first = evaluator.evaluate(criteria, experiences)
    second = evaluator.evaluate(criteria, experiences)
    assert first == second


def test_weighted_evaluator_penalizes_slow_recovery() -> None:
    fast = (
        _experience(SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED, 10.0),
        _experience(SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED, 10.0),
    )
    slow = (
        _experience(SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED, 90.0),
        _experience(SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED, 90.0),
    )
    evaluator = WeightedStrategyQualityEvaluator(recovery_time_reference_seconds=60.0)
    criteria = _criteria()
    fast_score = evaluator.evaluate(criteria, fast).quality_score
    slow_score = evaluator.evaluate(criteria, slow).quality_score
    assert fast_score > slow_score


def test_service_uses_repository_port_only() -> None:
    class RecordingRepository:
        def __init__(self) -> None:
            self.queries: list[tuple[str, str | None]] = []

        def append(
            self,
            experience: SelfHealingStrategyPerformanceExperience,
        ) -> SelfHealingStrategyPerformanceExperience:
            return experience

        def query(self, criteria: object) -> tuple[SelfHealingStrategyPerformanceExperience, ...]:
            from intergrax.contracts.self_healing.performance_memory.query import StrategyPerformanceMemoryQuery

            assert isinstance(criteria, StrategyPerformanceMemoryQuery)
            self.queries.append((criteria.tenant_id, criteria.strategy_id))
            return ()

    repo = RecordingRepository()
    service = StrategyQualityEvaluationService(
        repository=repo,
        evaluator=BasicStrategyQualityEvaluator(),
    )
    service.assess(_criteria())
    assert repo.queries == [(_TENANT_ID, _STRATEGY_ID)]
    assert isinstance(repo, StrategyPerformanceMemoryRepository)


def test_quality_evaluation_contracts_have_no_runtime_imports() -> None:
    assessment_module = importlib.import_module(
        "intergrax.contracts.self_healing.quality_evaluation.assessment",
    )
    evaluator_module = importlib.import_module(
        "intergrax.contracts.self_healing.quality_evaluation.evaluator",
    )
    for module in (assessment_module, evaluator_module):
        source_path = inspect.getfile(module)
        assert "intergrax\\runtime" not in source_path
        assert "intergrax/runtime" not in source_path


def test_assessment_rejects_invalid_outcome_totals() -> None:
    with pytest.raises(ValueError, match="outcome counts"):
        StrategyQualityAssessment(
            tenant_id=_TENANT_ID,
            strategy_id=_STRATEGY_ID,
            execution_count=3,
            successful_executions=2,
            failed_executions=0,
            rolled_back_executions=0,
            inconclusive_executions=0,
            success_ratio=1.0,
            average_recovery_time_seconds=1.0,
            min_recovery_time_seconds=1.0,
            max_recovery_time_seconds=1.0,
            quality_score=1.0,
            evaluator_id="test",
            evidence_refs=(),
        )
