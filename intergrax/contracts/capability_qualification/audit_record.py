# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed audit chain linking acquisition, qualification, and lifecycle (UCA-4)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_qualification.lifecycle_decision import (
    CapabilityQualificationLifecycleDecision,
    CapabilityQualificationLifecycleOutcome,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)

SCHEMA_CAPABILITY_QUALIFICATION_AUDIT_RECORD_V1: Final = (
    "capability_qualification_audit_record.v1"
)
_NON_EMPTY = Field(min_length=1)


class CapabilityQualificationAuditRecord(BaseModel):
    """Semantic provenance anchor — not application logging."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_qualification_audit_record.v1"] = (
        SCHEMA_CAPABILITY_QUALIFICATION_AUDIT_RECORD_V1
    )
    qualification_request_id: str = _NON_EMPTY
    acquisition_request_id: str = _NON_EMPTY
    acquisition_strategy_id: str = _NON_EMPTY
    gap_id: str = _NON_EMPTY
    qualification_provider_id: str | None = None
    qualification_outcome: CapabilityQualificationOutcome
    lifecycle_outcome: CapabilityQualificationLifecycleOutcome
    correlation_id: str | None = None
    causation_id: str | None = None

    @field_validator(
        "qualification_request_id",
        "acquisition_request_id",
        "acquisition_strategy_id",
        "gap_id",
        "qualification_provider_id",
        "correlation_id",
        "causation_id",
    )
    @classmethod
    def _validate_ids(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="id")


def build_qualification_audit_record(
    *,
    qualification_request_id: str,
    acquisition_request_id: str,
    acquisition_strategy_id: str,
    gap_id: str,
    qualification_provider_id: str | None,
    qualification_outcome: CapabilityQualificationOutcome,
    lifecycle_decision: CapabilityQualificationLifecycleDecision,
    correlation_id: str | None,
    causation_id: str | None,
) -> CapabilityQualificationAuditRecord:
    return CapabilityQualificationAuditRecord(
        qualification_request_id=qualification_request_id,
        acquisition_request_id=acquisition_request_id,
        acquisition_strategy_id=acquisition_strategy_id,
        gap_id=gap_id,
        qualification_provider_id=qualification_provider_id,
        qualification_outcome=qualification_outcome,
        lifecycle_outcome=lifecycle_decision.outcome,
        correlation_id=correlation_id,
        causation_id=causation_id,
    )


__all__ = [
    "SCHEMA_CAPABILITY_QUALIFICATION_AUDIT_RECORD_V1",
    "CapabilityQualificationAuditRecord",
    "build_qualification_audit_record",
]
