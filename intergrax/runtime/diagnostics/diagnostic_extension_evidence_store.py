# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Append-only extension evidence store (adapter-neutral, R5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from intergrax.contracts.diagnostic_extension_evidence import (
    DiagnosticEvidenceScope,
    DiagnosticExtensionEvidence,
    validate_extension_evidence_tenant_scope,
)


class DiagnosticExtensionEvidenceStore(Protocol):
    """Canonical extension evidence port — contributors never write Problems."""

    def append(self, evidence: DiagnosticExtensionEvidence) -> DiagnosticExtensionEvidence:
        """Persist one immutable evidence fact."""

    def query_for_scope(
        self,
        scope: DiagnosticEvidenceScope,
    ) -> tuple[DiagnosticExtensionEvidence, ...]:
        """Return evidence for one tenant-scoped execution scope."""


@dataclass(slots=True)
class InMemoryDiagnosticExtensionEvidenceStore:
    """Qualification / test store — deterministic ordering by evidence_id."""

    _records: list[DiagnosticExtensionEvidence] = field(default_factory=list)

    def append(self, evidence: DiagnosticExtensionEvidence) -> DiagnosticExtensionEvidence:
        validate_extension_evidence_tenant_scope(
            evidence,
            tenant_id=evidence.scope.tenant_id,
        )
        self._records.append(evidence)
        return evidence

    def query_for_scope(
        self,
        scope: DiagnosticEvidenceScope,
    ) -> tuple[DiagnosticExtensionEvidence, ...]:
        matched = [
            record
            for record in self._records
            if _scope_matches(record.scope, scope)
        ]
        return tuple(sorted(matched, key=lambda item: str(item.evidence_id)))


def _scope_matches(
    stored: DiagnosticEvidenceScope,
    query: DiagnosticEvidenceScope,
) -> bool:
    if stored.tenant_id != query.tenant_id:
        return False
    if stored.task_id != query.task_id:
        return False
    if stored.run_id != query.run_id:
        return False
    if query.attempt_id is not None and stored.attempt_id != query.attempt_id:
        return False
    if query.execution_id is not None and stored.execution_id != query.execution_id:
        return False
    return True


__all__ = [
    "DiagnosticExtensionEvidenceStore",
    "InMemoryDiagnosticExtensionEvidenceStore",
]
