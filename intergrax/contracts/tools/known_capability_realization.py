# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool domain handoff for known catalog capability realization (UCA-2)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_acquisition.success_evidence import (
    validate_availability_proves_host_available,
)
from intergrax.contracts.capability_catalog.evidence import (
    CapabilityDiscoveryAvailabilityEvidence,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.kind import CapabilityKind

SCHEMA_KNOWN_TOOL_CAPABILITY_REALIZATION_REQUEST_V1: Final = (
    "known_tool_capability_realization_request.v1"
)
SCHEMA_KNOWN_TOOL_CAPABILITY_REALIZATION_RESULT_V1: Final = (
    "known_tool_capability_realization_result.v1"
)
_NON_EMPTY = Field(min_length=1)


class KnownToolCapabilityRealizationOutcome(StrEnum):
    """Tool-owned realization terminal semantics — not UCA coordination outcome."""

    REALIZED = "realized"
    ALREADY_REALIZED = "already_realized"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    REQUIRES_HITL = "requires_hitl"


class KnownToolCapabilityRealizationRequest(BaseModel):
    """Tool lifecycle handoff for one known capability identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["known_tool_capability_realization_request.v1"] = (
        SCHEMA_KNOWN_TOOL_CAPABILITY_REALIZATION_REQUEST_V1
    )
    operation_id: str = _NON_EMPTY
    host_profile_id: str = _NON_EMPTY
    capability_identity: CapabilityIdentityKey
    requested_at: datetime

    @field_validator("operation_id", "host_profile_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("requested_at")
    @classmethod
    def _validate_requested_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("requested_at must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def _validate_tool_identity(self) -> KnownToolCapabilityRealizationRequest:
        if self.capability_identity.kind is not CapabilityKind.TOOL:
            raise ValueError(
                "known tool realization requires capability_identity.kind TOOL",
            )
        return self


class KnownToolCapabilityRealizationResult(BaseModel):
    """Tool-owned realization result with availability evidence slice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["known_tool_capability_realization_result.v1"] = (
        SCHEMA_KNOWN_TOOL_CAPABILITY_REALIZATION_RESULT_V1
    )
    outcome: KnownToolCapabilityRealizationOutcome
    operation_id: str = _NON_EMPTY
    host_profile_id: str = _NON_EMPTY
    capability_identity: CapabilityIdentityKey
    domain_reference: str | None = None
    availability_evidence: CapabilityDiscoveryAvailabilityEvidence | None = None
    reason_detail: str = ""

    @model_validator(mode="after")
    def _validate_success_evidence(self) -> KnownToolCapabilityRealizationResult:
        if self.capability_identity.kind is not CapabilityKind.TOOL:
            raise ValueError(
                "known tool realization result requires capability_identity.kind TOOL",
            )
        if self.outcome in (
            KnownToolCapabilityRealizationOutcome.REALIZED,
            KnownToolCapabilityRealizationOutcome.ALREADY_REALIZED,
        ):
            if self.availability_evidence is None:
                raise ValueError(
                    "REALIZED outcomes require availability evidence",
                )
            validate_availability_proves_host_available(
                identity=self.capability_identity,
                availability=self.availability_evidence,
            )
        return self


class KnownToolCapabilityRealizationPort(Protocol):
    """Public Tool domain realization boundary for known catalog identities."""

    def realize(
        self,
        request: KnownToolCapabilityRealizationRequest,
    ) -> KnownToolCapabilityRealizationResult: ...


__all__ = [
    "KnownToolCapabilityRealizationOutcome",
    "KnownToolCapabilityRealizationPort",
    "KnownToolCapabilityRealizationRequest",
    "KnownToolCapabilityRealizationResult",
    "SCHEMA_KNOWN_TOOL_CAPABILITY_REALIZATION_REQUEST_V1",
    "SCHEMA_KNOWN_TOOL_CAPABILITY_REALIZATION_RESULT_V1",
]
