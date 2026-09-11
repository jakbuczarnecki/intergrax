# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable typed extension evidence (DIAG extension SPI R5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
    mint_event_id,
)

DIAGNOSTIC_EXTENSION_EVIDENCE_SCHEMA: Literal["diagnostic_extension_evidence.v1"] = (
    "diagnostic_extension_evidence.v1"
)


@dataclass(frozen=True, slots=True)
class DiagnosticEvidenceScope:
    """Bounded execution scope for extension evidence and analysis."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticExecutionContext:
    """Bounded extension execution boundary — no global state or custom ContextVar."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    execution_id: ExecutionId | None
    evidence_scope: DiagnosticEvidenceScope
    time_budget_ms: int


@dataclass(frozen=True, slots=True)
class DiagnosticEvidenceContext:
    """Input to evidence contributors — alias view over execution context."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    execution_id: ExecutionId | None
    time_budget_ms: int

    @staticmethod
    def from_execution_context(
        context: DiagnosticExecutionContext,
    ) -> DiagnosticEvidenceContext:
        return DiagnosticEvidenceContext(
            tenant_id=context.tenant_id,
            task_id=context.task_id,
            run_id=context.run_id,
            attempt_id=context.attempt_id,
            execution_id=context.execution_id,
            time_budget_ms=context.time_budget_ms,
        )

    def to_evidence_scope(self) -> DiagnosticEvidenceScope:
        return DiagnosticEvidenceScope(
            tenant_id=self.tenant_id,
            task_id=self.task_id,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            execution_id=self.execution_id,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticExtensionEvidence:
    """One immutable contributed evidence fact — not a Problem or root-cause claim."""

    schema_version: Literal["diagnostic_extension_evidence.v1"]
    evidence_id: EventId
    scope: DiagnosticEvidenceScope
    evidence_namespace: str
    kind: str
    summary: str
    detail_ref: str | None = None

    @staticmethod
    def mint(
        *,
        scope: DiagnosticEvidenceScope,
        evidence_namespace: str,
        kind: str,
        summary: str,
        detail_ref: str | None = None,
        evidence_id: EventId | None = None,
    ) -> DiagnosticExtensionEvidence:
        namespace = _require_namespaced_token(evidence_namespace, label="evidence_namespace")
        kind_token = _require_namespaced_token(kind, label="kind")
        if not summary.strip():
            raise ValueError("summary is required")
        return DiagnosticExtensionEvidence(
            schema_version=DIAGNOSTIC_EXTENSION_EVIDENCE_SCHEMA,
            evidence_id=evidence_id or mint_event_id(),
            scope=scope,
            evidence_namespace=namespace,
            kind=kind_token,
            summary=summary.strip(),
            detail_ref=detail_ref,
        )


def validate_extension_evidence_tenant_scope(
    evidence: DiagnosticExtensionEvidence,
    *,
    tenant_id: str,
) -> None:
    if evidence.scope.tenant_id != tenant_id:
        raise ValueError("extension evidence tenant_id does not match execution scope")


def _require_namespaced_token(value: str, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be str")
    normalized = value.strip()
    if not normalized or "." not in normalized:
        raise ValueError(f"{label} must be a namespaced token (e.g. company.domain.kind)")
    if normalized != value.strip():
        raise ValueError(f"{label} must not contain leading or trailing whitespace")
    return normalized


__all__ = [
    "DIAGNOSTIC_EXTENSION_EVIDENCE_SCHEMA",
    "DiagnosticEvidenceContext",
    "DiagnosticEvidenceScope",
    "DiagnosticExecutionContext",
    "DiagnosticExtensionEvidence",
    "validate_extension_evidence_tenant_scope",
]
