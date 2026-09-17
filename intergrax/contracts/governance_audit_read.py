# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only governance audit/decision facts (GV-01 read seam)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_runtime_governance import GovernanceAuditEvent
from intergrax.contracts.execution_identity import ExecutionId


@runtime_checkable
class GovernanceAuditReadPort(Protocol):
    """Returns recorded governance decisions for an execution — no re-evaluation."""

    @property
    def source_id(self) -> str: ...

    def list_audit_events_for_execution(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        limit: int,
    ) -> tuple[GovernanceAuditEvent, ...]: ...


__all__ = ["GovernanceAuditReadPort"]
