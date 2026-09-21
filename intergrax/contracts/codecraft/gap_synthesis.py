# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Gap-anchored CodeCraft synthesis — build artifact after canonical CapabilityGap (UCA-6A)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.need import CapabilityNeed

SCHEMA_CODECRAFT_GAP_SYNTHESIS_REQUEST_V1: Final = "codecraft_gap_synthesis_request.v1"
SCHEMA_CODECRAFT_GAP_SYNTHESIS_RESULT_V1: Final = "codecraft_gap_synthesis_result.v1"


class CodeCraftGapSynthesisOutcome(StrEnum):
    """CodeCraft domain outcome — not UCA qualification or production execution."""

    SUCCEEDED = "succeeded"
    NOT_SUPPORTED = "not_supported"
    UNAVAILABLE = "unavailable"
    BLOCKED = "blocked"
    REQUIRES_HITL = "requires_hitl"
    FAILED = "failed"


class CodeCraftGapSynthesisRequest(BaseModel):
    """Synthesize a capability artifact for an existing CapabilityGap.

    Does not re-run canonical platform capability discovery (UCA-1).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["codecraft_gap_synthesis_request.v1"] = (
        SCHEMA_CODECRAFT_GAP_SYNTHESIS_REQUEST_V1
    )
    operation_id: str = Field(min_length=1)
    gap_id: str = Field(min_length=1)
    canonical_discovery_correlation_id: str = Field(min_length=1)
    capability_need: CapabilityNeed
    synthesis_goal: str = Field(min_length=1)
    synthesis_constraints: str = ""
    target_kind: CapabilityKind = CapabilityKind.TOOL
    correlation_id: str | None = None
    causation_id: str | None = None

    @field_validator(
        "operation_id",
        "gap_id",
        "canonical_discovery_correlation_id",
        "synthesis_goal",
        "correlation_id",
        "causation_id",
    )
    @classmethod
    def _validate_ids(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label=str(info.field_name))


class CodeCraftGapSynthesisResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["codecraft_gap_synthesis_result.v1"] = (
        SCHEMA_CODECRAFT_GAP_SYNTHESIS_RESULT_V1
    )
    operation_id: str = Field(min_length=1)
    gap_id: str = Field(min_length=1)
    outcome: CodeCraftGapSynthesisOutcome
    artifact_reference: str | None = None
    domain_handoff_reference: str | None = None
    codecraft_operation_correlation_id: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    reason_detail: str = ""

    @field_validator("operation_id", "gap_id")
    @classmethod
    def _validate_required_ids(cls, value: str, info) -> str:
        return require_non_empty_text(value, label=str(info.field_name))

    @model_validator(mode="after")
    def _succeeded_requires_acquisition_subject(self) -> CodeCraftGapSynthesisResult:
        if self.outcome is not CodeCraftGapSynthesisOutcome.SUCCEEDED:
            return self
        if not self.domain_handoff_reference and not self.artifact_reference:
            raise ValueError(
                "SUCCEEDED requires domain_handoff_reference or artifact_reference",
            )
        return self


@runtime_checkable
class CodeCraftGapSynthesisPort(Protocol):
    """Public CodeCraft seam for UCA gap synthesis — replaceable implementation."""

    def synthesize_from_gap(
        self,
        request: CodeCraftGapSynthesisRequest,
    ) -> CodeCraftGapSynthesisResult:
        """Build or package a synthesized capability artifact when technically possible."""
        ...


__all__ = [
    "CodeCraftGapSynthesisOutcome",
    "CodeCraftGapSynthesisPort",
    "CodeCraftGapSynthesisRequest",
    "CodeCraftGapSynthesisResult",
    "SCHEMA_CODECRAFT_GAP_SYNTHESIS_REQUEST_V1",
    "SCHEMA_CODECRAFT_GAP_SYNTHESIS_RESULT_V1",
]
