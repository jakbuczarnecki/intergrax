# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Immutable runtime fact snapshot for intelligence analyzers (W6-B)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.runtime_intelligence.errors import InvalidIntelligenceContextError

RUNTIME_INTELLIGENCE_CONTEXT_SCHEMA_VERSION = "runtime_intelligence_context.v1"


class RuntimeIntelligenceFactKind(StrEnum):
    """Observed fact families referenced by context — pointers only."""

    RUNTIME_EVENT = "runtime_event"
    CHECKPOINT = "checkpoint"
    TERMINAL = "terminal"
    LINEAGE = "lineage"
    ADMISSION_AUDIT = "admission_audit"
    RECOVERY_RECORD = "recovery_record"
    CANCEL_RECORD = "cancel_record"


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceFactReference:
    fact_kind: RuntimeIntelligenceFactKind
    fact_ref: str

    def __post_init__(self) -> None:
        if not self.fact_ref.strip():
            raise ValueError("fact_ref must be non-empty")


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceContextMetadata:
    collected_at: datetime
    correlation_id: str
    schema_version: str = RUNTIME_INTELLIGENCE_CONTEXT_SCHEMA_VERSION
    context_label: str = ""

    def __post_init__(self) -> None:
        if not self.correlation_id.strip():
            raise ValueError("correlation_id must be non-empty")
        if not self.schema_version.strip():
            raise ValueError("schema_version must be non-empty")


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceContext:
    """
    Read-only execution snapshot.

    Ownership: built by runtime integration / projection layer; consumed by analyzers.
    """

    tenant_id: str
    task_id: str
    run_id: str
    fact_references: tuple[RuntimeIntelligenceFactReference, ...]
    metadata: RuntimeIntelligenceContextMetadata
    attempt_id: str | None = None
    execution_id: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")


def validate_runtime_intelligence_context(context: RuntimeIntelligenceContext) -> None:
    """Raise InvalidIntelligenceContextError when the snapshot is unusable for analysis."""
    if not context.fact_references:
        raise InvalidIntelligenceContextError("fact_references must be non-empty for analysis")
    if context.metadata.collected_at.tzinfo is None:
        raise InvalidIntelligenceContextError("metadata.collected_at must be timezone-aware")


__all__ = [
    "RUNTIME_INTELLIGENCE_CONTEXT_SCHEMA_VERSION",
    "RuntimeIntelligenceContext",
    "RuntimeIntelligenceContextMetadata",
    "RuntimeIntelligenceFactKind",
    "RuntimeIntelligenceFactReference",
    "validate_runtime_intelligence_context",
]
