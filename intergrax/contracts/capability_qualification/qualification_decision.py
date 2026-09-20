# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Combined qualification coordination output (UCA-4)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from intergrax.contracts.capability_qualification.audit_record import (
    CapabilityQualificationAuditRecord,
)
from intergrax.contracts.capability_qualification.lifecycle_decision import (
    CapabilityQualificationLifecycleDecision,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)

SCHEMA_CAPABILITY_QUALIFICATION_DECISION_V1: Final = (
    "capability_qualification_decision.v1"
)


class CapabilityQualificationDecision(BaseModel):
    """Qualification result with lifecycle disposition and audit chain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_qualification_decision.v1"] = (
        SCHEMA_CAPABILITY_QUALIFICATION_DECISION_V1
    )
    qualification_result: CapabilityQualificationResult
    lifecycle_decision: CapabilityQualificationLifecycleDecision
    audit_record: CapabilityQualificationAuditRecord

    @model_validator(mode="after")
    def _validate_internal_consistency(self) -> CapabilityQualificationDecision:
        result = self.qualification_result
        audit = self.audit_record
        lifecycle = self.lifecycle_decision
        if audit.qualification_request_id != result.qualification_request_id:
            raise ValueError(
                "audit qualification_request_id must match qualification_result",
            )
        if audit.acquisition_request_id != result.acquisition_request_id:
            raise ValueError(
                "audit acquisition_request_id must match qualification_result",
            )
        if audit.acquisition_strategy_id != result.strategy_id:
            raise ValueError(
                "audit acquisition_strategy_id must match result strategy_id"
            )
        if audit.gap_id != result.gap_id:
            raise ValueError("audit gap_id must match qualification_result")
        if audit.qualification_provider_id != result.provider_id:
            raise ValueError(
                "audit qualification_provider_id must match result provider_id",
            )
        if audit.qualification_outcome != result.outcome:
            raise ValueError("audit qualification_outcome must match result outcome")
        if audit.lifecycle_outcome != lifecycle.outcome:
            raise ValueError("audit lifecycle_outcome must match lifecycle_decision")
        if audit.correlation_id != result.correlation_id:
            raise ValueError("audit correlation_id must match qualification_result")
        if audit.causation_id != result.causation_id:
            raise ValueError("audit causation_id must match qualification_result")
        return self


__all__ = [
    "SCHEMA_CAPABILITY_QUALIFICATION_DECISION_V1",
    "CapabilityQualificationDecision",
]
