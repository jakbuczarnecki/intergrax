# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory adapter for ``AutonomyExecutionAuditRepository`` (SELF-HEALING R6.3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.self_healing.autonomy.execution_audit import AutonomyExecutionAuditRecord


@dataclass
class InMemoryAutonomyExecutionAuditRepository:
    records: list[AutonomyExecutionAuditRecord] = field(default_factory=list)

    def append(self, record: AutonomyExecutionAuditRecord) -> AutonomyExecutionAuditRecord:
        self.records.append(record)
        return record


__all__ = ["InMemoryAutonomyExecutionAuditRepository"]
