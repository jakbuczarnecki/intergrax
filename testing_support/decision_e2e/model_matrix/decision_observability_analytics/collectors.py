# © Artur Czarnecki. All rights reserved.

"""Observation collector plugins (DS-E2E-15J-L8)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from testing_support.decision_e2e.model_matrix.decision_observability_analytics.contracts import (
    DecisionObservation,
    DecisionObservationMetadata,
    ObservationEventKind,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleEvent,
    DecisionLifecycleRecord,
    DecisionLifecycleState,
    DecisionSourceReference,
    DecisionType,
)
from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.contracts import (
    GovernanceDisposition,
)


@dataclass(frozen=True, slots=True)
class LifecycleObservationCollector:
    """Build observations from lifecycle records and transition events."""

    records: tuple[DecisionLifecycleRecord, ...]
    events: tuple[DecisionLifecycleEvent, ...] = ()
    collector_id: str = "lifecycle"

    def collect(self) -> tuple[DecisionObservation, ...]:
        record_by_id = {item.decision_id: item for item in self.records}
        observations: list[DecisionObservation] = []
        for record in self.records:
            observations.append(
                DecisionObservation(
                    decision_id=record.decision_id,
                    lifecycle_state=record.lifecycle_state,
                    decision_type=record.decision_type,
                    source_references=record.source_references,
                    metadata=DecisionObservationMetadata(
                        event_kind=ObservationEventKind.LIFECYCLE_RECORD,
                        observed_at=record.created_at,
                    ),
                )
            )
        for event in self.events:
            record = record_by_id.get(event.decision_id)
            if record is None:
                continue
            observations.append(
                DecisionObservation(
                    decision_id=event.decision_id,
                    lifecycle_state=event.new_state,
                    decision_type=record.decision_type,
                    source_references=record.source_references,
                    metadata=DecisionObservationMetadata(
                        event_kind=ObservationEventKind.LIFECYCLE_TRANSITION,
                        observed_at=event.timestamp,
                        transition_previous_state=event.previous_state,
                        transition_new_state=event.new_state,
                        transition_reason=event.reason,
                    ),
                )
            )
        return tuple(observations)


@dataclass(frozen=True, slots=True)
class GovernanceFactInput:
    decision_id: str
    lifecycle_state: DecisionLifecycleState
    decision_type: DecisionType
    source_references: tuple[DecisionSourceReference, ...]
    disposition: GovernanceDisposition
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class GovernanceFactObservationCollector:
    """Emit governance disposition observations without evaluating policy."""

    facts: tuple[GovernanceFactInput, ...]
    collector_id: str = "governance_facts"

    def collect(self) -> tuple[DecisionObservation, ...]:
        return tuple(
            DecisionObservation(
                decision_id=fact.decision_id,
                lifecycle_state=fact.lifecycle_state,
                decision_type=fact.decision_type,
                source_references=fact.source_references,
                metadata=DecisionObservationMetadata(
                    event_kind=ObservationEventKind.GOVERNANCE_OUTCOME,
                    observed_at=fact.observed_at,
                    governance_disposition=fact.disposition,
                ),
            )
            for fact in self.facts
        )


@dataclass(frozen=True, slots=True)
class StaticObservationCollector:
    """Test or custom collector that returns pre-built observations."""

    observations: tuple[DecisionObservation, ...]
    collector_id: str = "static"

    def collect(self) -> tuple[DecisionObservation, ...]:
        return self.observations


__all__ = [
    "GovernanceFactInput",
    "GovernanceFactObservationCollector",
    "LifecycleObservationCollector",
    "StaticObservationCollector",
]
