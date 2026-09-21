# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Qualification provider selection policy contracts (UCA-4)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_qualification.provider_descriptor import (
    CapabilityQualificationProviderDescriptor,
)
from intergrax.contracts.capability_qualification.qualification_reason_code import (
    CapabilityQualificationReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
)

SCHEMA_CAPABILITY_QUALIFICATION_PROVIDER_SELECTION_V1: Final = (
    "capability_qualification_provider_selection.v1"
)
_NON_EMPTY = Field(min_length=1)


class CapabilityQualificationProviderSelectionOutcome(StrEnum):
    """Typed provider selection — not first-wins."""

    SELECTED = "selected"
    NO_PROVIDER = "no_provider"
    CONFLICT = "conflict"


class CapabilityQualificationProviderSelection(BaseModel):
    """Deterministic provider selection outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_qualification_provider_selection.v1"] = (
        SCHEMA_CAPABILITY_QUALIFICATION_PROVIDER_SELECTION_V1
    )
    outcome: CapabilityQualificationProviderSelectionOutcome
    provider_id: str | None = None
    reason_code: CapabilityQualificationReasonCode
    reason_detail: str = ""

    @field_validator("provider_id")
    @classmethod
    def _validate_provider_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="provider_id")

    @model_validator(mode="after")
    def _outcome_provider_id_invariant(
        self,
    ) -> CapabilityQualificationProviderSelection:
        if self.outcome is CapabilityQualificationProviderSelectionOutcome.SELECTED:
            if self.provider_id is None:
                raise ValueError("SELECTED outcome requires provider_id")
            if self.reason_code is not CapabilityQualificationReasonCode.NONE:
                raise ValueError("SELECTED outcome requires reason_code NONE")
        elif self.provider_id is not None:
            raise ValueError("provider_id is only allowed when outcome is SELECTED")
        return self


@runtime_checkable
class CapabilityQualificationProviderSelectionPolicy(Protocol):
    """Pluginable provider selection — orchestrator must not embed preferences."""

    def select(
        self,
        *,
        request: CapabilityQualificationRequest,
        candidates: tuple[CapabilityQualificationProviderDescriptor, ...],
    ) -> CapabilityQualificationProviderSelection:
        """Choose at most one provider from eligible candidates."""
        ...


__all__ = [
    "CapabilityQualificationProviderSelection",
    "CapabilityQualificationProviderSelectionOutcome",
    "CapabilityQualificationProviderSelectionPolicy",
    "SCHEMA_CAPABILITY_QUALIFICATION_PROVIDER_SELECTION_V1",
]
