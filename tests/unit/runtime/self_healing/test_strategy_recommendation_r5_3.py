# © Artur Czarnecki. All rights reserved.

"""SELF-HEALING R5.3 strategy recommendation engine foundation."""

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
from intergrax.contracts.self_healing.quality_evaluation import StrategyQualityEvaluationCriteria
from intergrax.contracts.self_healing.strategy_recommendation import (
    StrategyRecommendationBasisKind,
    StrategyRecommendationConfidenceLevel,
    StrategyRecommendationEngine,
    StrategyRecommendationRequest,
)
from intergrax.runtime.self_healing.performance_memory import InMemoryStrategyPerformanceMemoryRepository
from intergrax.runtime.self_healing.quality_evaluation import (
    BasicStrategyQualityEvaluator,
    StrategyQualityEvaluationService,
    WeightedStrategyQualityEvaluator,
)
from intergrax.runtime.self_healing.strategy_recommendation import (
    QualityBasedStrategyRecommendationEngine,
    StrategyRecommendationService,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT_ID = "tenant-a"
_INVESTIGATION_ID = "inv-db-connection"
_PROBLEM_ID = "problem.database.connection_failure"
_RETRY = "platform.database.retry_connection"
_RESTART = "platform.database.restart_pool"
_FAILOVER = "platform.database.failover_connection"


def _experience(
    strategy_id: str,
    outcome: SelfHealingStrategyExecutionOutcome,
    recovery_seconds: float,
    *,
    evidence_suffix: str,
) -> SelfHealingStrategyPerformanceExperience:
    return SelfHealingStrategyPerformanceExperience(
        experience_id=mint_self_healing_strategy_performance_experience_id(),
        tenant_id=_TENANT_ID,
        strategy_id=strategy_id,
        workflow_id="sh_wf_rec00000001",
        plan_id="sh_plan_rec000001",
        execution_ids=("exec-1",),
        diagnostic_investigation_id=_INVESTIGATION_ID,
        execution_outcome=outcome,
        rollback_executed=outcome is SelfHealingStrategyExecutionOutcome.ROLLED_BACK,
        recovery_time_seconds=recovery_seconds,
        evidence_refs=(f"evidence://run/{strategy_id}/{evidence_suffix}",),
        recorded_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )


def _request(*strategy_ids: str) -> StrategyRecommendationRequest:
    return StrategyRecommendationRequest(
        tenant_id=_TENANT_ID,
        diagnostic_investigation_id=_INVESTIGATION_ID,
        problem_id=_PROBLEM_ID,
        candidate_strategy_ids=strategy_ids,
    )


def _service(
    repo: InMemoryStrategyPerformanceMemoryRepository,
    *,
    weighted: bool = False,
) -> StrategyRecommendationService:
    evaluator = (
        WeightedStrategyQualityEvaluator(recovery_time_reference_seconds=60.0)
        if weighted
        else BasicStrategyQualityEvaluator()
    )
    quality = StrategyQualityEvaluationService(repository=repo, evaluator=evaluator)
    return StrategyRecommendationService(
        quality_evaluation=quality,
        engine=QualityBasedStrategyRecommendationEngine(),
    )


def test_recommendation_engine_protocol_is_runtime_checkable() -> None:
    engine = QualityBasedStrategyRecommendationEngine()
    assert isinstance(engine, StrategyRecommendationEngine)


def test_recommends_highest_historical_quality_among_candidates() -> None:
    repo = InMemoryStrategyPerformanceMemoryRepository()
    for row in (
        _experience(_RETRY, SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED, 5.0, evidence_suffix="a"),
        _experience(_RETRY, SelfHealingStrategyExecutionOutcome.REPAIR_FAILED, 5.0, evidence_suffix="b"),
        _experience(_RESTART, SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED, 4.0, evidence_suffix="c"),
        _experience(_RESTART, SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED, 4.0, evidence_suffix="d"),
        _experience(_RESTART, SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED, 4.0, evidence_suffix="e"),
        _experience(_FAILOVER, SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED, 3.0, evidence_suffix="f"),
    ):
        repo.append(row)
    recommendation = _service(repo).recommend(_request(_RETRY, _RESTART))
    assert recommendation.recommended_strategy_id == _RESTART
    assert recommendation.ranked_strategy_ids == (_RESTART, _RETRY)
    assert recommendation.basis.kind is StrategyRecommendationBasisKind.HIGHEST_HISTORICAL_QUALITY_SCORE
    assert recommendation.basis.summary == "Highest historical quality score"
    assert recommendation.confidence is StrategyRecommendationConfidenceLevel.HIGH
    assert recommendation.supporting_assessment.quality_score == pytest.approx(1.0)


def test_no_historical_data_yields_insufficient_data_confidence() -> None:
    recommendation = _service(InMemoryStrategyPerformanceMemoryRepository()).recommend(
        _request(_RESTART, _RETRY),
    )
    assert recommendation.recommended_strategy_id == _RESTART
    assert recommendation.confidence is StrategyRecommendationConfidenceLevel.INSUFFICIENT_DATA
    assert recommendation.basis.kind is StrategyRecommendationBasisKind.INSUFFICIENT_HISTORICAL_EVIDENCE
    assert recommendation.supporting_assessment.execution_count == 0


def test_quality_score_tie_uses_stable_strategy_id_ordering() -> None:
    repo = InMemoryStrategyPerformanceMemoryRepository()
    for strategy_id in (_RETRY, _RESTART):
        repo.append(
            _experience(
                strategy_id,
                SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED,
                10.0,
                evidence_suffix="x",
            ),
        )
    recommendation = _service(repo).recommend(_request(_RESTART, _RETRY))
    assert recommendation.recommended_strategy_id == _RESTART
    assert recommendation.basis.kind is StrategyRecommendationBasisKind.QUALITY_SCORE_TIE_BREAKER
    assert recommendation.confidence is StrategyRecommendationConfidenceLevel.LOW


def test_weighted_evaluator_changes_quality_score_without_duplicating_logic() -> None:
    repo = InMemoryStrategyPerformanceMemoryRepository()
    repo.append(
        _experience(
            _RETRY,
            SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED,
            90.0,
            evidence_suffix="slow",
        ),
    )
    basic_rec = _service(repo).recommend(_request(_RETRY))
    weighted_rec = _service(repo, weighted=True).recommend(_request(_RETRY))
    assert basic_rec.recommended_strategy_id == _RETRY
    assert weighted_rec.recommended_strategy_id == _RETRY
    assert weighted_rec.supporting_assessment.evaluator_id != basic_rec.supporting_assessment.evaluator_id
    assert weighted_rec.supporting_assessment.quality_score < basic_rec.supporting_assessment.quality_score


def test_service_uses_quality_evaluation_port_only() -> None:
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
    quality = StrategyQualityEvaluationService(
        repository=repo,
        evaluator=BasicStrategyQualityEvaluator(),
    )
    service = StrategyRecommendationService(
        quality_evaluation=quality,
        engine=QualityBasedStrategyRecommendationEngine(),
    )
    service.recommend(_request(_RETRY, _RESTART))
    assert repo.queries == [
        (_TENANT_ID, _RETRY),
        (_TENANT_ID, _RESTART),
    ]
    assert isinstance(repo, StrategyPerformanceMemoryRepository)


def test_recommendation_contracts_have_no_runtime_imports() -> None:
    modules = (
        "intergrax.contracts.self_healing.strategy_recommendation.recommendation",
        "intergrax.contracts.self_healing.strategy_recommendation.engine",
    )
    for module_name in modules:
        module = importlib.import_module(module_name)
        source_path = inspect.getfile(module)
        assert "intergrax\\runtime" not in source_path
        assert "intergrax/runtime" not in source_path


def test_recommendation_runtime_has_no_execution_coupling() -> None:
    forbidden_tokens = (
        "lifecycle",
        "orchestrator",
        "SelfHealingActionProvider",
        "execution_engine",
        "HighestConfidenceStrategySelector",
    )
    module_names = (
        "intergrax.runtime.self_healing.strategy_recommendation.service",
        "intergrax.runtime.self_healing.strategy_recommendation.quality_based_engine",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        source = inspect.getsource(module)
        lowered = source.lower()
        for token in forbidden_tokens:
            assert token.lower() not in lowered


def test_recommendation_request_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="candidate_strategy_ids"):
        StrategyRecommendationRequest(
            tenant_id=_TENANT_ID,
            diagnostic_investigation_id=_INVESTIGATION_ID,
            problem_id=_PROBLEM_ID,
            candidate_strategy_ids=(),
        )


def test_quality_evaluation_criteria_unchanged_for_recommendation_reads() -> None:
    criteria = StrategyQualityEvaluationCriteria(tenant_id=_TENANT_ID, strategy_id=_RETRY)
    assert criteria.strategy_id == _RETRY
