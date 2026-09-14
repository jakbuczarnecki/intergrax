# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Immutable ERL reliability diagnostic observation contract."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.enterprise_reliability.case_lifecycle import (
    ReliabilityCaseLifecycleState,
)
from intergrax.contracts.enterprise_reliability.diagnostics.artifact_refs import (
    ReliabilityDiagnosticArtifactRefs,
)
from intergrax.contracts.enterprise_reliability.diagnostics.correlation import (
    ReliabilityDiagnosticCorrelation,
)
from intergrax.contracts.enterprise_reliability.diagnostics.taxonomy import (
    AutomationSafetyHint,
    ExternalEffectReliabilitySignalKind,
)

SCHEMA_EXTERNAL_EFFECT_RELIABILITY_OBSERVATION_V1: Final = (
    "external_effect_reliability_observation.v1"
)
MAX_RELIABILITY_DIAGNOSTIC_TRACE_REFS: Final = 16
_MAX_TRACE_REF_LEN: Final = 512

_SIGNAL_KINDS_REQUIRING_EVIDENCE_REF: Final[frozenset[ExternalEffectReliabilitySignalKind]] = (
    frozenset(
        {
            ExternalEffectReliabilitySignalKind.RECONCILIATION_ATTEMPTED,
            ExternalEffectReliabilitySignalKind.EVIDENCE_INSUFFICIENT,
            ExternalEffectReliabilitySignalKind.EVIDENCE_SUFFICIENT,
        },
    )
)


class ExternalEffectReliabilityObservationValidationError(ValueError):
    """Malformed or inconsistent reliability diagnostic observation."""


class ExternalEffectReliabilityObservation(BaseModel):
    """One material, audit-friendly ERL fact for operator diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external_effect_reliability_observation.v1"] = (
        SCHEMA_EXTERNAL_EFFECT_RELIABILITY_OBSERVATION_V1
    )
    observation_id: str = Field(min_length=1, max_length=256)
    tenant_id: str = Field(min_length=1, max_length=128)
    signal_kind: ExternalEffectReliabilitySignalKind
    recorded_at: datetime
    reliability_case_id: str = Field(min_length=1, max_length=256)
    correlation: ReliabilityDiagnosticCorrelation
    lifecycle_state: ReliabilityCaseLifecycleState
    artifact_refs: ReliabilityDiagnosticArtifactRefs
    execution_safety_hint: AutomationSafetyHint
    trace_refs: tuple[str, ...] = ()
    source_transition_id: str | None = Field(default=None, max_length=512)

    @field_validator("observation_id", mode="after")
    @classmethod
    def _observation_id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("observation_id must not be whitespace-only")
        return value

    @field_validator("trace_refs", mode="after")
    @classmethod
    def _validate_trace_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > MAX_RELIABILITY_DIAGNOSTIC_TRACE_REFS:
            raise ValueError(
                f"trace_refs exceeds max {MAX_RELIABILITY_DIAGNOSTIC_TRACE_REFS}",
            )
        for index, ref in enumerate(value):
            if not ref or not ref.strip():
                raise ValueError(f"trace_refs[{index}] must be non-empty")
            if len(ref) > _MAX_TRACE_REF_LEN:
                raise ValueError(f"trace_refs[{index}] exceeds max length")
        return value

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> ExternalEffectReliabilityObservation:
        if self.tenant_id != self.correlation.tenant_id:
            raise ExternalEffectReliabilityObservationValidationError(
                "tenant_id must match correlation.tenant_id",
            )
        if self.reliability_case_id != self.correlation.reliability_case_id:
            raise ExternalEffectReliabilityObservationValidationError(
                "reliability_case_id must match correlation.reliability_case_id",
            )
        if self.signal_kind in _SIGNAL_KINDS_REQUIRING_EVIDENCE_REF:
            evidence = self.artifact_refs.evidence_ref
            if evidence is None or not evidence.strip():
                raise ExternalEffectReliabilityObservationValidationError(
                    "artifact_refs.evidence_ref required for signal_kind "
                    f"{self.signal_kind.value}",
                )
        return self


__all__ = [
    "ExternalEffectReliabilityObservation",
    "ExternalEffectReliabilityObservationValidationError",
    "MAX_RELIABILITY_DIAGNOSTIC_TRACE_REFS",
    "SCHEMA_EXTERNAL_EFFECT_RELIABILITY_OBSERVATION_V1",
]
