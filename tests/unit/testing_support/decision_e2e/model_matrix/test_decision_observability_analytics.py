# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from testing_support.decision_e2e.model_matrix.decision_observability_analytics import (
    OBSERVABILITY_TASK_ID,
    AnalysisPayloadKind,
    AnalyticsResultStatus,
    CustomAnalysisPayload,
    DecisionAnalyticsAuditMetadata,
    DecisionAnalyticsResult,
    DecisionObservabilityEngine,
    DecisionObservation,
    DecisionOutcomeAnalyzer,
    GovernanceFactInput,
    GovernanceFactObservationCollector,
    LifecycleObservationCollector,
    ObservationEventKind,
    StaticObservationCollector,
    default_decision_observability_engine,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleActorRef,
    DecisionLifecycleEvent,
    DecisionLifecycleRecord,
    DecisionLifecycleState,
    DecisionSourceKind,
    DecisionSourceReference,
    DecisionType,
)
from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.contracts import (
    GovernanceDisposition as GovDisposition,
)


def _stamp(offset_seconds: int = 0) -> datetime:
    return datetime(2026, 9, 12, 10, 0, 0, tzinfo=UTC) + timedelta(
        seconds=offset_seconds
    )


def _record(
    decision_id: str,
    *,
    state: DecisionLifecycleState = DecisionLifecycleState.COMPLETED,
) -> DecisionLifecycleRecord:
    return DecisionLifecycleRecord(
        decision_id=decision_id,
        decision_type=DecisionType.PRODUCTION_MODEL_ROUTING,
        lifecycle_state=state,
        created_at=_stamp(),
        source_references=(
            DecisionSourceReference(
                source_kind=DecisionSourceKind.ORCHESTRATION,
                reference_id="orch-1",
            ),
        ),
    )


def test_lifecycle_collector_builds_factual_observation() -> None:
    record = _record("decision-a")
    collector = LifecycleObservationCollector(records=(record,))

    observations = collector.collect()

    assert len(observations) == 1
    observation = observations[0]
    assert observation.decision_id == "decision-a"
    assert observation.lifecycle_state is DecisionLifecycleState.COMPLETED
    assert observation.decision_type is DecisionType.PRODUCTION_MODEL_ROUTING
    assert observation.metadata.event_kind is ObservationEventKind.LIFECYCLE_RECORD
    assert observation.metadata.observed_at == record.created_at


def test_outcome_analyzer_over_multiple_observations() -> None:
    record_completed = _record("d1", state=DecisionLifecycleState.COMPLETED)
    record_failed = _record("d2", state=DecisionLifecycleState.FAILED)
    record_blocked = _record("d3", state=DecisionLifecycleState.REJECTED)
    collector = LifecycleObservationCollector(
        records=(record_completed, record_failed, record_blocked)
    )
    engine = default_decision_observability_engine(
        collectors=(collector,),
        analyzers=(DecisionOutcomeAnalyzer(),),
    )

    result = engine.run(run_at=_stamp(30))

    assert result.observability_task_id == OBSERVABILITY_TASK_ID
    assert len(result.analytics_results) == 1
    analytics = result.analytics_results[0]
    assert analytics.status is AnalyticsResultStatus.COMPLETE
    assert analytics.payload_kind is AnalysisPayloadKind.OUTCOME
    assert analytics.outcome_payload is not None
    assert analytics.outcome_payload.completed == 1
    assert analytics.outcome_payload.failed == 1
    assert analytics.outcome_payload.blocked == 1
    assert analytics.audit.analyzer_id == "decision_outcome"
    assert analytics.audit.analyzer_version == "1"
    assert set(analytics.audit.decision_ids) == {"d1", "d2", "d3"}


class _TestTokenAnalyzer:
    analyzer_id = "test_token"
    analyzer_version = "test-1"

    def analyze(
        self,
        observations: tuple[DecisionObservation, ...],
        *,
        analyzed_at: datetime,
    ) -> DecisionAnalyticsResult:
        if not observations:
            return DecisionAnalyticsResult(
                observability_task_id=OBSERVABILITY_TASK_ID,
                status=AnalyticsResultStatus.INSUFFICIENT_DATA,
                audit=DecisionAnalyticsAuditMetadata(
                    analyzer_id=self.analyzer_id,
                    analyzer_version=self.analyzer_version,
                    decision_ids=(),
                    period_start=None,
                    period_end=None,
                ),
                payload_kind=AnalysisPayloadKind.CUSTOM,
            )
        stamps = [item.metadata.observed_at for item in observations]
        return DecisionAnalyticsResult(
            observability_task_id=OBSERVABILITY_TASK_ID,
            status=AnalyticsResultStatus.COMPLETE,
            audit=DecisionAnalyticsAuditMetadata(
                analyzer_id=self.analyzer_id,
                analyzer_version=self.analyzer_version,
                decision_ids=tuple(
                    dict.fromkeys(item.decision_id for item in observations)
                ),
                period_start=min(stamps),
                period_end=max(stamps),
            ),
            payload_kind=AnalysisPayloadKind.CUSTOM,
            custom_payload=CustomAnalysisPayload(
                summary_token="plugin_ok",
                numeric_fact=len(observations),
            ),
        )


def test_custom_analyzer_plugin_without_engine_changes() -> None:
    record = _record("plugin-decision")
    engine = DecisionObservabilityEngine(
        collectors=(LifecycleObservationCollector(records=(record,)),),
        analyzers=(_TestTokenAnalyzer(),),
        metrics_providers=(),
        report_providers=(),
    )

    run = engine.run(run_at=_stamp(40))

    custom = run.analytics_results[0]
    assert custom.audit.analyzer_id == "test_token"
    assert custom.custom_payload is not None
    assert custom.custom_payload.summary_token == "plugin_ok"


class _EmptyCollector:
    collector_id = "empty"

    def collect(self) -> tuple[DecisionObservation, ...]:
        return ()


def test_swappable_collector_implementation() -> None:
    record = _record("swap-me")
    lifecycle_engine = default_decision_observability_engine(
        collectors=(LifecycleObservationCollector(records=(record,)),),
        analyzers=(DecisionOutcomeAnalyzer(),),
    )
    static_engine = default_decision_observability_engine(
        collectors=(
            StaticObservationCollector(
                observations=lifecycle_engine.collectors[0].collect()
            ),
        ),
        analyzers=(DecisionOutcomeAnalyzer(),),
    )

    lifecycle_run = lifecycle_engine.run(run_at=_stamp(50))
    static_run = static_engine.run(run_at=_stamp(50))

    assert lifecycle_run.observations == static_run.observations
    assert (
        lifecycle_run.analytics_results[0].outcome_payload
        == static_run.analytics_results[0].outcome_payload
    )


def test_no_observations_returns_controlled_results() -> None:
    engine = default_decision_observability_engine(
        collectors=(_EmptyCollector(),),
        analyzers=(DecisionOutcomeAnalyzer(),),
    )

    run = engine.run(run_at=_stamp(60))

    assert run.observations == ()
    assert run.analytics_results
    assert all(
        item.status is AnalyticsResultStatus.INSUFFICIENT_DATA
        for item in run.analytics_results
    )


def test_governance_analyzer_counts_dispositions() -> None:
    fact = GovernanceFactInput(
        decision_id="gov-1",
        lifecycle_state=DecisionLifecycleState.APPROVED,
        decision_type=DecisionType.PRODUCTION_MODEL_ROUTING,
        source_references=(
            DecisionSourceReference(
                source_kind=DecisionSourceKind.GOVERNANCE,
                reference_id="gov-ref",
            ),
        ),
        disposition=GovDisposition.BLOCK,
        observed_at=_stamp(70),
    )
    engine = default_decision_observability_engine(
        collectors=(GovernanceFactObservationCollector(facts=(fact,)),),
    )
    run = engine.run(run_at=_stamp(80))
    governance_result = next(
        item
        for item in run.analytics_results
        if item.payload_kind is AnalysisPayloadKind.GOVERNANCE
    )
    assert governance_result.status is AnalyticsResultStatus.COMPLETE
    assert governance_result.governance_payload is not None
    assert governance_result.governance_payload.block == 1


def test_lifecycle_performance_from_transition_events() -> None:
    record = _record("perf-1", state=DecisionLifecycleState.CREATED)
    events = (
        DecisionLifecycleEvent(
            decision_id="perf-1",
            previous_state=DecisionLifecycleState.CREATED,
            new_state=DecisionLifecycleState.COMPLETED,
            reason="done",
            timestamp=_stamp(3),
            actor=DecisionLifecycleActorRef(actor_kind="test", reference_id="t"),
        ),
    )
    collector = LifecycleObservationCollector(records=(record,), events=events)
    engine = default_decision_observability_engine(collectors=(collector,))
    run = engine.run(run_at=_stamp(90))
    performance = next(
        item
        for item in run.analytics_results
        if item.payload_kind is AnalysisPayloadKind.LIFECYCLE_PERFORMANCE
    )
    assert performance.status is AnalyticsResultStatus.COMPLETE
    assert performance.performance_payload is not None
    assert performance.performance_payload.average_transition_seconds == 3.0
