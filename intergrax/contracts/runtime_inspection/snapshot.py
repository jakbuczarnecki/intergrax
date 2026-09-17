# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Canonical federated runtime inspection snapshot (INSPECT-01-A)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness
from intergrax.contracts.runtime_inspection.failures import RuntimeInspectionSourceFailure
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionContinuationSection,
    RuntimeInspectionDiagnosticSection,
    RuntimeInspectionEvidenceSection,
    RuntimeInspectionExecutionStateSection,
    RuntimeInspectionGovernanceSection,
    RuntimeInspectionIdentitySection,
    RuntimeInspectionTimelineSection,
    RuntimeInspectionToolSection,
)


class RuntimeInspectionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "runtime_inspection_snapshot.v1"
    observed_at: datetime
    identity: RuntimeInspectionIdentitySection
    execution: RuntimeInspectionExecutionStateSection
    timeline: RuntimeInspectionTimelineSection
    diagnostics: RuntimeInspectionDiagnosticSection | None = None
    evidence: RuntimeInspectionEvidenceSection | None = None
    tools: RuntimeInspectionToolSection | None = None
    governance: RuntimeInspectionGovernanceSection | None = None
    continuation: RuntimeInspectionContinuationSection | None = None
    completeness: RuntimeInspectionCompleteness
    source_failures: tuple[RuntimeInspectionSourceFailure, ...] = Field(default_factory=tuple)
    consistency_note: str = Field(
        default="snapshot_consistent_for_configured_sources",
        min_length=1,
    )

    @field_validator("observed_at")
    @classmethod
    def _validate_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        return value


__all__ = ["RuntimeInspectionSnapshot"]
