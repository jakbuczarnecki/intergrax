# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only governance audit access."""

from __future__ import annotations

from intergrax.contracts.agent_runtime_governance import GovernanceAuditEvent
from intergrax.contracts.execution_identity import ExecutionId
from intergrax.contracts.governance_audit_read import GovernanceAuditReadPort
from intergrax.runtime.agent_governance.audit import InMemoryGovernanceAuditSink


class InMemoryGovernanceAuditReadAdapter:
    """GovernanceAuditReadPort over an injected in-memory audit sink."""

    def __init__(
        self,
        sink: InMemoryGovernanceAuditSink,
        *,
        source_id: str = "governance_audit_memory",
    ) -> None:
        self._sink = sink
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    def list_audit_events_for_execution(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        limit: int,
    ) -> tuple[GovernanceAuditEvent, ...]:
        matched: list[GovernanceAuditEvent] = []
        for event in self._sink.events:
            if event.execution_id != execution_id:
                continue
            matched.append(event)
            if len(matched) >= limit:
                break
        matched.sort(key=lambda item: (item.timestamp, item.event_id))
        return tuple(matched)


__all__ = ["InMemoryGovernanceAuditReadAdapter"]
