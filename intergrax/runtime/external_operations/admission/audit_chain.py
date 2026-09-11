# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Append-only external operation audit chain (R1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from intergrax.contracts.external_operations.attempt import ExternalOperationAttempt
from intergrax.contracts.external_operations.audit import ExternalOperationAuditRecord
from intergrax.contracts.external_operations.admission import OperationAdmissionDecision


@runtime_checkable
class ExternalOperationAuditChain(Protocol):
    def append(
        self,
        *,
        attempt: ExternalOperationAttempt,
        admission: OperationAdmissionDecision,
        actor: str,
        evidence_refs: tuple[str, ...] = (),
    ) -> ExternalOperationAuditRecord:
        ...

    def reconstruct(self, attempt_id: str) -> list[ExternalOperationAuditRecord]:
        ...


class InMemoryExternalOperationAuditChain:
    def __init__(self) -> None:
        self._records: list[ExternalOperationAuditRecord] = []

    def append(
        self,
        *,
        attempt: ExternalOperationAttempt,
        admission: OperationAdmissionDecision,
        actor: str,
        evidence_refs: tuple[str, ...] = (),
    ) -> ExternalOperationAuditRecord:
        record = ExternalOperationAuditRecord(
            attempt_id=attempt.attempt_id,
            intent_id=attempt.intent.intent_id,
            tenant_id=attempt.intent.tenant_id,
            provider_id=attempt.provider_id,
            admission_decision=admission,
            execution_status=attempt.lifecycle,
            recorded_at=datetime.now(timezone.utc),
            actor=actor,
            evidence_refs=evidence_refs,
        )
        self._records.append(record)
        return record

    def reconstruct(self, attempt_id: str) -> list[ExternalOperationAuditRecord]:
        return [r for r in self._records if r.attempt_id == attempt_id]
