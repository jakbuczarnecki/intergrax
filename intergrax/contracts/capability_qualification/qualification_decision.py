# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Combined qualification coordination output (UCA-4)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

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


__all__ = [
    "SCHEMA_CAPABILITY_QUALIFICATION_DECISION_V1",
    "CapabilityQualificationDecision",
]
