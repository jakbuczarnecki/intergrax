# © Artur Czarnecki. All rights reserved.

"""Decision lifecycle orchestration via injected providers (DS-E2E-15J-L7)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleActorRef,
    DecisionLifecycleEvent,
    DecisionLifecycleRecord,
    DecisionLifecycleState,
    DecisionSourceReference,
    DecisionType,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.protocol import (
    DecisionAuditProvider,
    DecisionClockProvider,
    DecisionIdentityProvider,
    DecisionStateTransitionProvider,
)


@dataclass(frozen=True, slots=True)
class DecisionLifecycleEngine:
    transition_provider: DecisionStateTransitionProvider
    audit_provider: DecisionAuditProvider
    clock_provider: DecisionClockProvider
    identity_provider: DecisionIdentityProvider

    def begin_decision(
        self,
        *,
        decision_type: DecisionType,
        source_references: tuple[DecisionSourceReference, ...],
        actor: DecisionLifecycleActorRef,
        reason: str,
    ) -> DecisionLifecycleRecord:
        created_at = self.clock_provider.now()
        record = DecisionLifecycleRecord(
            decision_id=self.identity_provider.new_decision_id(),
            decision_type=decision_type,
            lifecycle_state=DecisionLifecycleState.CREATED,
            created_at=created_at,
            source_references=source_references,
        )
        self.audit_provider.record_lifecycle_event(
            DecisionLifecycleEvent(
                decision_id=record.decision_id,
                previous_state=None,
                new_state=DecisionLifecycleState.CREATED,
                reason=reason,
                timestamp=created_at,
                actor=actor,
            )
        )
        return record

    def transition(
        self,
        record: DecisionLifecycleRecord,
        *,
        to_state: DecisionLifecycleState,
        reason: str,
        actor: DecisionLifecycleActorRef,
    ) -> DecisionLifecycleRecord:
        self.transition_provider.assert_transition_allowed(
            from_state=record.lifecycle_state,
            to_state=to_state,
        )
        timestamp = self.clock_provider.now()
        self.audit_provider.record_lifecycle_event(
            DecisionLifecycleEvent(
                decision_id=record.decision_id,
                previous_state=record.lifecycle_state,
                new_state=to_state,
                reason=reason,
                timestamp=timestamp,
                actor=actor,
            )
        )
        return DecisionLifecycleRecord(
            decision_id=record.decision_id,
            decision_type=record.decision_type,
            lifecycle_state=to_state,
            created_at=record.created_at,
            source_references=record.source_references,
        )


__all__ = ["DecisionLifecycleEngine"]
