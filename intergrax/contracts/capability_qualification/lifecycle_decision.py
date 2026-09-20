# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Post-qualification lifecycle decision contracts — decision only (UCA-4)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)

SCHEMA_CAPABILITY_QUALIFICATION_LIFECYCLE_DECISION_V1: Final = (
    "capability_qualification_lifecycle_decision.v1"
)
_NON_EMPTY = Field(min_length=1)


class CapabilityQualificationLifecycleOutcome(StrEnum):
    """What happens next — not domain lifecycle execution."""

    ACCEPT = "accept"
    REJECT = "reject"
    QUARANTINE = "quarantine"
    REQUIRES_HITL = "requires_hitl"
    RETAIN_FOR_REVIEW = "retain_for_review"
    DISCARD = "discard"


class CapabilityQualificationLifecycleReasonCode(StrEnum):
    """Typed lifecycle decision reasons."""

    NONE = "none"
    QUALIFICATION_ACCEPTED = "qualification_accepted"
    QUALIFICATION_REJECTED = "qualification_rejected"
    QUALIFICATION_BLOCKED = "qualification_blocked"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_FAILED = "provider_failed"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    SELECTION_CONFLICT = "selection_conflict"
    NO_QUALIFICATION_PROVIDER = "no_qualification_provider"


class CapabilityQualificationLifecycleDecision(BaseModel):
    """Immutable lifecycle disposition — no registry mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_qualification_lifecycle_decision.v1"] = (
        SCHEMA_CAPABILITY_QUALIFICATION_LIFECYCLE_DECISION_V1
    )
    outcome: CapabilityQualificationLifecycleOutcome
    reason_code: CapabilityQualificationLifecycleReasonCode
    reason_detail: str = ""

    @field_validator("reason_detail")
    @classmethod
    def _validate_reason_detail(cls, value: str) -> str:
        if value == "":
            return ""
        return require_non_empty_text(value, label="reason_detail")


@runtime_checkable
class CapabilityQualificationLifecyclePolicy(Protocol):
    """Pluginable lifecycle disposition from qualification facts."""

    def decide(
        self,
        *,
        request: CapabilityQualificationRequest,
        qualification_result: CapabilityQualificationResult,
    ) -> CapabilityQualificationLifecycleDecision:
        """Return typed next-step disposition — no domain side effects."""
        ...


__all__ = [
    "CapabilityQualificationLifecycleDecision",
    "CapabilityQualificationLifecycleOutcome",
    "CapabilityQualificationLifecyclePolicy",
    "CapabilityQualificationLifecycleReasonCode",
    "SCHEMA_CAPABILITY_QUALIFICATION_LIFECYCLE_DECISION_V1",
]
