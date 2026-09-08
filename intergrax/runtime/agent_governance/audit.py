# © Artur Czarnecki. All rights reserved.

"""Governance audit event recording."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.agent_runtime_governance import (
    GovernanceAuditEvent,
    GovernanceAuditSinkPort,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    mint_governance_audit_event_id,
)


class InMemoryGovernanceAuditSink:
    """Test-friendly audit sink with explicit injected storage."""

    def __init__(self) -> None:
        self._events: list[GovernanceAuditEvent] = []

    @property
    def events(self) -> tuple[GovernanceAuditEvent, ...]:
        return tuple(self._events)

    def record(self, event: GovernanceAuditEvent) -> None:
        self._events.append(event)


class GovernanceAuditRecorder:
    """Builds and emits governance audit events for every decision."""

    def __init__(self, sink: GovernanceAuditSinkPort) -> None:
        self._sink = sink

    def record_decision(
        self,
        request: ToolAuthorizationRequest,
        decision: ToolAuthorizationDecision,
        *,
        timestamp: datetime | None = None,
    ) -> GovernanceAuditEvent:
        event = GovernanceAuditEvent(
            event_id=mint_governance_audit_event_id(),
            execution_id=request.execution_id,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            task_id=request.task_id,
            agent_id=request.agent.agent_id,
            capability=request.capability,
            tool_id=request.tool_id,
            decision=decision.decision,
            policy_results=decision.policy_results,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        self._sink.record(event)
        return event
