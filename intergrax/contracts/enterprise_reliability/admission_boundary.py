# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ERL admission boundary contracts — domain-neutral entry from external effects."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.case_lifecycle import (
    ReliabilityCaseLifecycleRecord,
)
from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectContract,
    UnknownUncertaintyPosture,
)
from intergrax.contracts.enterprise_reliability.lifecycle import UncertaintyStateRecord
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.enterprise_reliability.reliability_boundary import (
    ExternalEffectReliabilityProjection,
)

SCHEMA_EXTERNAL_EFFECT_ADMISSION_BOUNDARY_V1: Final = (
    "external_effect_admission_boundary.v1"
)


class ExternalEffectAdmissionPhase(StrEnum):
    """Admission pipeline position — not a second reliability case lifecycle."""

    RECEIVED = "received"
    CLASSIFIED = "classified"
    CASE_CREATED = "case_created"
    HANDED_OFF = "handed_off"


class ExternalEffectAdmissionContextError(ValueError):
    """Admission request is incomplete or inconsistent for ERL entry."""


class ExternalEffectAdmissionCaseError(RuntimeError):
    """Reliability case could not be initialized for an admitted uncertainty episode."""


class ExternalEffectAdmissionSourceContext(BaseModel):
    """Opaque caller context — no domain business payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_kind: str = Field(min_length=1, max_length=128)
    source_ref: str = Field(min_length=1, max_length=512)


class ExternalEffectAdmissionRequest(BaseModel):
    """Domain-neutral external effect reliability context at ERL entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    external_effect_ref: str = Field(min_length=1, max_length=512)
    effect_outcome: ExternalEffectOutcome
    correlation_id: str = Field(min_length=1, max_length=256)
    contract: ExternalEffectContract
    source_context: ExternalEffectAdmissionSourceContext
    case_id: str | None = Field(default=None, max_length=256)
    uncertainty_state_ref: str | None = Field(default=None, max_length=512)
    reason: str = Field(default="", max_length=512)


class ExternalEffectAdmissionResult(BaseModel):
    """Admission outcome linking external effect truth to optional reliability case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: ExternalEffectAdmissionPhase
    external_effect_ref: str = Field(min_length=1, max_length=512)
    correlation_id: str = Field(min_length=1, max_length=256)
    contract_id: str = Field(min_length=1, max_length=256)
    projection: ExternalEffectReliabilityProjection
    uncertainty_state: UncertaintyStateRecord | None = None
    unknown_posture: UnknownUncertaintyPosture | None = None
    case_record: ReliabilityCaseLifecycleRecord | None = None


def uncertainty_state_ref_for_correlation(correlation_id: str) -> str:
    """Stable platform ref for an admitted uncertainty episode."""
    return f"erl:uncertainty:{correlation_id}"


def reliability_case_id_for_admission(
    *,
    correlation_id: str,
    contract_id: str,
) -> str:
    """Stable platform case identity for one correlation and effect contract."""
    return f"erl:case:{correlation_id}:{contract_id}"


def assert_external_effect_admission_request(
    request: ExternalEffectAdmissionRequest,
) -> None:
    """Fail closed before runtime admission orchestration."""
    if not request.correlation_id.strip():
        raise ExternalEffectAdmissionContextError("missing correlation_id")
    if not request.external_effect_ref.strip():
        raise ExternalEffectAdmissionContextError("missing external_effect_ref")
    if request.case_id is not None and not request.case_id.strip():
        raise ExternalEffectAdmissionContextError("invalid case_id")
    if request.uncertainty_state_ref is not None:
        ref = request.uncertainty_state_ref.strip()
        if not ref:
            raise ExternalEffectAdmissionContextError("invalid uncertainty_state_ref")
        expected = uncertainty_state_ref_for_correlation(request.correlation_id)
        if ref != expected:
            raise ExternalEffectAdmissionContextError(
                "uncertainty_state_ref must match platform correlation ref",
            )


__all__ = [
    "ExternalEffectAdmissionCaseError",
    "ExternalEffectAdmissionContextError",
    "ExternalEffectAdmissionPhase",
    "ExternalEffectAdmissionRequest",
    "ExternalEffectAdmissionResult",
    "ExternalEffectAdmissionSourceContext",
    "SCHEMA_EXTERNAL_EFFECT_ADMISSION_BOUNDARY_V1",
    "assert_external_effect_admission_request",
    "reliability_case_id_for_admission",
    "uncertainty_state_ref_for_correlation",
]
