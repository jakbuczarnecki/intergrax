# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only federation source ports for runtime inspection (INSPECT-01-A)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.contracts.execution_reconstruction import ExecutionReconstruction
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionDiagnosticSection,
    RuntimeInspectionEvidenceSection,
)


class RuntimeInspectionScopeLookupOutcome(StrEnum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    TENANT_DENIED = "tenant_denied"


@dataclass(frozen=True, slots=True)
class RuntimeInspectionExecutionScope:
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    execution_id: ExecutionId


@dataclass(frozen=True, slots=True)
class RuntimeInspectionScopeLookupResult:
    outcome: RuntimeInspectionScopeLookupOutcome
    scope: RuntimeInspectionExecutionScope | None = None


@runtime_checkable
class RuntimeInspectionExecutionScopeReader(Protocol):
    """Resolve canonical execution identity for inspection — no minting."""

    @property
    def source_id(self) -> str: ...

    def resolve_scope(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> RuntimeInspectionScopeLookupResult: ...


@runtime_checkable
class RuntimeInspectionExecutionFactsReader(Protocol):
    """Required canonical execution reconstruction facts."""

    @property
    def source_id(self) -> str: ...

    def read_execution_facts(
        self,
        scope: RuntimeInspectionExecutionScope,
    ) -> ExecutionReconstruction: ...


@runtime_checkable
class RuntimeInspectionDiagnosticReadPort(Protocol):
    """Optional diagnostic interpretation for one execution scope."""

    @property
    def source_id(self) -> str: ...

    def read_diagnostics(
        self,
        scope: RuntimeInspectionExecutionScope,
    ) -> RuntimeInspectionDiagnosticSection: ...


@runtime_checkable
class RuntimeInspectionEvidenceReadPort(Protocol):
    """Optional evidence references for one execution scope."""

    @property
    def source_id(self) -> str: ...

    def read_evidence_references(
        self,
        scope: RuntimeInspectionExecutionScope,
    ) -> RuntimeInspectionEvidenceSection: ...


__all__ = [
    "RuntimeInspectionDiagnosticReadPort",
    "RuntimeInspectionEvidenceReadPort",
    "RuntimeInspectionExecutionFactsReader",
    "RuntimeInspectionExecutionScope",
    "RuntimeInspectionExecutionScopeReader",
    "RuntimeInspectionScopeLookupOutcome",
    "RuntimeInspectionScopeLookupResult",
]
