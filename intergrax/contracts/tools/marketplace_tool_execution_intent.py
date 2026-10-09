# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical Marketplace Tool execution intent — source-neutral common truth + typed provenance."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey

SCHEMA_MARKETPLACE_TOOL_EXECUTION_INTENT_V2: Final = "marketplace_tool_execution_intent.v2"


class MarketplaceToolExecutionProvenanceKind(StrEnum):
    UCA = "uca"
    CONFIGURED = "configured"


class UcaMarketplaceToolExecutionProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provenance_kind: Literal[MarketplaceToolExecutionProvenanceKind.UCA] = (
        MarketplaceToolExecutionProvenanceKind.UCA
    )
    handoff_id: str = Field(min_length=1)
    resume_operation_id: str = Field(min_length=1)
    uca_qualified_subject_reference: str = Field(min_length=1)

    @field_validator(
        "handoff_id",
        "resume_operation_id",
        "uca_qualified_subject_reference",
    )
    @classmethod
    def _validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return require_non_empty_text(value, label=str(info.field_name))


class ConfiguredMarketplaceToolExecutionProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provenance_kind: Literal[MarketplaceToolExecutionProvenanceKind.CONFIGURED] = (
        MarketplaceToolExecutionProvenanceKind.CONFIGURED
    )
    recovery_decision_id: str = Field(min_length=1)
    acquisition_decision_id: str = Field(min_length=1)
    configured_binding_operation_id: str = Field(min_length=1)
    configured_execution_operation_id: str = Field(min_length=1)
    configuration_adoption_identity: str = Field(min_length=1)

    @field_validator(
        "recovery_decision_id",
        "acquisition_decision_id",
        "configured_binding_operation_id",
        "configured_execution_operation_id",
        "configuration_adoption_identity",
    )
    @classmethod
    def _validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return require_non_empty_text(value, label=str(info.field_name))


MarketplaceToolExecutionProvenance = Annotated[
    UcaMarketplaceToolExecutionProvenance | ConfiguredMarketplaceToolExecutionProvenance,
    Field(discriminator="provenance_kind"),
]


class MarketplaceToolExecutionIntent(BaseModel):
    """Durable pre-EE Tool execution intent — one capability identity, closed provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_tool_execution_intent.v2"] = (
        SCHEMA_MARKETPLACE_TOOL_EXECUTION_INTENT_V2
    )
    execution_request_id: str = Field(min_length=1)
    binding_operation_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    worker_need_id: str = Field(min_length=1)
    subject_reference: str = Field(min_length=1)
    capability_identity: CapabilityIdentityKey
    selected_operation: str = Field(min_length=1)
    execution_target_correlation: str | None = None
    provenance: MarketplaceToolExecutionProvenance

    @field_validator(
        "execution_request_id",
        "binding_operation_id",
        "tenant_id",
        "task_id",
        "worker_need_id",
        "subject_reference",
        "selected_operation",
    )
    @classmethod
    def _validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return require_non_empty_text(value, label=str(info.field_name))

    @field_validator("execution_target_correlation")
    @classmethod
    def _validate_optional_correlation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="execution_target_correlation")

    @property
    def qualified_subject_reference(self) -> str:
        """Legacy binding-subject handle naming — same as ``subject_reference``."""
        return self.subject_reference


__all__ = [
    "ConfiguredMarketplaceToolExecutionProvenance",
    "MarketplaceToolExecutionIntent",
    "MarketplaceToolExecutionProvenance",
    "MarketplaceToolExecutionProvenanceKind",
    "SCHEMA_MARKETPLACE_TOOL_EXECUTION_INTENT_V2",
    "UcaMarketplaceToolExecutionProvenance",
]
