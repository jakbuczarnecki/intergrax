# © Artur Czarnecki. All rights reserved.

"""Built-in decision analytics analyzers (extend via new classes)."""

from __future__ import annotations

from datetime import datetime

from testing_support.decision_e2e.model_matrix.decision_observability_analytics.contracts import (
    OBSERVABILITY_TASK_ID,
    AnalysisPayloadKind,
    AnalyticsResultStatus,
    DecisionAnalyticsAuditMetadata,
    DecisionAnalyticsResult,
    DecisionObservation,
    DecisionOutcomeCounts,
    GovernanceOutcomeCounts,
    LifecyclePerformancePayload,
    ObservationEventKind,
    TransitionDurationFact,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleState as LifecycleState,
)
from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.contracts import (
    GovernanceDisposition,
)


def _audit_from_observations(
    *,
    analyzer_id: str,
    analyzer_version: str,
    observations: tuple[DecisionObservation, ...],
) -> DecisionAnalyticsAuditMetadata:
    if not observations:
        return DecisionAnalyticsAuditMetadata(
            analyzer_id=analyzer_id,
            analyzer_version=analyzer_version,
            decision_ids=(),
            period_start=None,
            period_end=None,
        )
    decision_ids = tuple(dict.fromkeys(item.decision_id for item in observations))
    stamps = [item.metadata.observed_at for item in observations]
    return DecisionAnalyticsAuditMetadata(
        analyzer_id=analyzer_id,
        analyzer_version=analyzer_version,
        decision_ids=decision_ids,
        period_start=min(stamps),
        period_end=max(stamps),
    )


def _latest_state_by_decision(
    observations: tuple[DecisionObservation, ...],
) -> dict[str, LifecycleState]:
    latest: dict[str, tuple[datetime, LifecycleState]] = {}
    for item in observations:
        current = latest.get(item.decision_id)
        if current is None or item.metadata.observed_at >= current[0]:
            latest[item.decision_id] = (item.metadata.observed_at, item.lifecycle_state)
    return {key: value[1] for key, value in latest.items()}


def _insufficient_result(
    *,
    analyzer_id: str,
    analyzer_version: str,
    payload_kind: AnalysisPayloadKind,
) -> DecisionAnalyticsResult:
    return DecisionAnalyticsResult(
        observability_task_id=OBSERVABILITY_TASK_ID,
        status=AnalyticsResultStatus.INSUFFICIENT_DATA,
        audit=DecisionAnalyticsAuditMetadata(
            analyzer_id=analyzer_id,
            analyzer_version=analyzer_version,
            decision_ids=(),
            period_start=None,
            period_end=None,
        ),
        payload_kind=payload_kind,
    )


class DecisionOutcomeAnalyzer:
    analyzer_id = "decision_outcome"
    analyzer_version = "1"

    def analyze(
        self,
        observations: tuple[DecisionObservation, ...],
        *,
        analyzed_at: datetime,
    ) -> DecisionAnalyticsResult:
        del analyzed_at
        if not observations:
            return _insufficient_result(
                analyzer_id=self.analyzer_id,
                analyzer_version=self.analyzer_version,
                payload_kind=AnalysisPayloadKind.OUTCOME,
            )
        latest = _latest_state_by_decision(observations)
        completed = 0
        failed = 0
        blocked = 0
        in_progress = 0
        for state in latest.values():
            if state is LifecycleState.COMPLETED:
                completed += 1
            elif state is LifecycleState.FAILED:
                failed += 1
            elif state is LifecycleState.REJECTED:
                blocked += 1
            else:
                in_progress += 1
        return DecisionAnalyticsResult(
            observability_task_id=OBSERVABILITY_TASK_ID,
            status=AnalyticsResultStatus.COMPLETE,
            audit=_audit_from_observations(
                analyzer_id=self.analyzer_id,
                analyzer_version=self.analyzer_version,
                observations=observations,
            ),
            payload_kind=AnalysisPayloadKind.OUTCOME,
            outcome_payload=DecisionOutcomeCounts(
                completed=completed,
                failed=failed,
                blocked=blocked,
                in_progress=in_progress,
            ),
        )


class LifecyclePerformanceAnalyzer:
    analyzer_id = "lifecycle_performance"
    analyzer_version = "1"

    def analyze(
        self,
        observations: tuple[DecisionObservation, ...],
        *,
        analyzed_at: datetime,
    ) -> DecisionAnalyticsResult:
        del analyzed_at
        transitions = [
            item
            for item in observations
            if item.metadata.event_kind is ObservationEventKind.LIFECYCLE_TRANSITION
            and item.metadata.transition_previous_state is not None
        ]
        if not transitions:
            return _insufficient_result(
                analyzer_id=self.analyzer_id,
                analyzer_version=self.analyzer_version,
                payload_kind=AnalysisPayloadKind.LIFECYCLE_PERFORMANCE,
            )
        all_by_decision: dict[str, list[DecisionObservation]] = {}
        for item in observations:
            all_by_decision.setdefault(item.decision_id, []).append(item)
        duration_facts: list[TransitionDurationFact] = []
        for decision_id, items in all_by_decision.items():
            ordered = sorted(items, key=lambda row: row.metadata.observed_at)
            for index, current in enumerate(ordered):
                if (
                    current.metadata.event_kind
                    is not ObservationEventKind.LIFECYCLE_TRANSITION
                ):
                    continue
                prior = ordered[index - 1] if index > 0 else None
                if prior is None:
                    continue
                delta = (
                    current.metadata.observed_at - prior.metadata.observed_at
                ).total_seconds()
                duration_facts.append(
                    TransitionDurationFact(
                        decision_id=decision_id,
                        previous_state=current.metadata.transition_previous_state,
                        new_state=current.metadata.transition_new_state
                        or current.lifecycle_state,
                        duration_seconds=delta,
                    )
                )
        average: float | None = None
        if duration_facts:
            average = sum(item.duration_seconds for item in duration_facts) / len(
                duration_facts
            )
        return DecisionAnalyticsResult(
            observability_task_id=OBSERVABILITY_TASK_ID,
            status=AnalyticsResultStatus.COMPLETE,
            audit=_audit_from_observations(
                analyzer_id=self.analyzer_id,
                analyzer_version=self.analyzer_version,
                observations=observations,
            ),
            payload_kind=AnalysisPayloadKind.LIFECYCLE_PERFORMANCE,
            performance_payload=LifecyclePerformancePayload(
                transition_durations=tuple(duration_facts),
                average_transition_seconds=average,
            ),
        )


class GovernanceOutcomeAnalyzer:
    analyzer_id = "governance_outcome"
    analyzer_version = "1"

    def analyze(
        self,
        observations: tuple[DecisionObservation, ...],
        *,
        analyzed_at: datetime,
    ) -> DecisionAnalyticsResult:
        del analyzed_at
        governance_observations = [
            item
            for item in observations
            if item.metadata.governance_disposition is not None
        ]
        if not governance_observations:
            return _insufficient_result(
                analyzer_id=self.analyzer_id,
                analyzer_version=self.analyzer_version,
                payload_kind=AnalysisPayloadKind.GOVERNANCE,
            )
        allow = 0
        block = 0
        require_approval = 0
        for item in governance_observations:
            disposition = item.metadata.governance_disposition
            if disposition is GovernanceDisposition.ALLOW:
                allow += 1
            elif disposition is GovernanceDisposition.BLOCK:
                block += 1
            elif disposition is GovernanceDisposition.REQUIRE_APPROVAL:
                require_approval += 1
        return DecisionAnalyticsResult(
            observability_task_id=OBSERVABILITY_TASK_ID,
            status=AnalyticsResultStatus.COMPLETE,
            audit=_audit_from_observations(
                analyzer_id=self.analyzer_id,
                analyzer_version=self.analyzer_version,
                observations=tuple(governance_observations),
            ),
            payload_kind=AnalysisPayloadKind.GOVERNANCE,
            governance_payload=GovernanceOutcomeCounts(
                allow=allow,
                block=block,
                require_approval=require_approval,
            ),
        )


def default_decision_analytics_analyzers() -> tuple[
    DecisionOutcomeAnalyzer,
    LifecyclePerformanceAnalyzer,
    GovernanceOutcomeAnalyzer,
]:
    return (
        DecisionOutcomeAnalyzer(),
        LifecyclePerformanceAnalyzer(),
        GovernanceOutcomeAnalyzer(),
    )


__all__ = [
    "DecisionOutcomeAnalyzer",
    "GovernanceOutcomeAnalyzer",
    "LifecyclePerformanceAnalyzer",
    "default_decision_analytics_analyzers",
]
