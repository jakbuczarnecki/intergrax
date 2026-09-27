# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable marketplace-qualified Tool staging contracts (S24-GAP-02-P1)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.capability_catalog.release_identity import (
    CapabilityReleaseIdentity,
)
from intergrax.contracts.marketplace.handoff_traceability import (
    CapabilityHandoffConsumerTarget,
)

SCHEMA_MARKETPLACE_QUALIFIED_TOOL_STAGE_V1: Final = (
    "marketplace_qualified_tool_stage.v1"
)


class MarketplaceQualifiedToolStageWriteOutcome(StrEnum):
    """Terminal write semantics for Tool-owned marketplace staging."""

    CREATED = "created"
    ALREADY_STAGED_IDENTICAL = "already_staged_identical"


class MarketplaceQualifiedToolStageWriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: MarketplaceQualifiedToolStageWriteOutcome


class MarketplaceQualifiedToolStageError(Exception):
    """Base error for Tool marketplace qualified staging."""


class MarketplaceQualifiedToolStageConflictError(MarketplaceQualifiedToolStageError):
    """Same handoff identity was already staged with a different immutable payload."""


class MarketplaceQualifiedToolStageUnavailableError(MarketplaceQualifiedToolStageError):
    """Staging backend or dependency is unavailable."""


class MarketplaceQualifiedToolStageIntegrityError(MarketplaceQualifiedToolStageError):
    """Persistent staging record cannot be reconstructed into the canonical stage."""


class MarketplaceQualifiedToolStage(BaseModel):
    """Inactive, exact Tool release staged for later qualification — not executable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_qualified_tool_stage.v1"] = (
        SCHEMA_MARKETPLACE_QUALIFIED_TOOL_STAGE_V1
    )
    handoff_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    selected_release: CapabilityReleaseIdentity
    discovery_correlation_id: str = Field(min_length=1)
    selection_id: str = Field(min_length=1)
    consumer_target: CapabilityHandoffConsumerTarget
    downstream_consumer_id: str = Field(min_length=1)
    recorded_at: datetime

    @field_validator(
        "handoff_id",
        "tenant_id",
        "discovery_correlation_id",
        "selection_id",
        "downstream_consumer_id",
    )
    @classmethod
    def _validate_non_empty_ids(cls, value: str, info: ValidationInfo) -> str:
        return require_non_empty_text(value, label=str(info.field_name))

    @model_validator(mode="after")
    def _validate_tool_staging_invariants(self) -> MarketplaceQualifiedToolStage:
        if self.consumer_target is not CapabilityHandoffConsumerTarget.TOOL_DOMAIN:
            raise ValueError(
                "consumer_target must be TOOL_DOMAIN for marketplace qualified tool stage",
            )
        if self.selected_release.discovery.kind is not CapabilityKind.TOOL:
            raise ValueError(
                "selected_release must identify a TOOL capability",
            )
        if self.recorded_at.tzinfo is None:
            raise ValueError("recorded_at must be timezone-aware")
        return self


@runtime_checkable
class MarketplaceQualifiedToolStageRepository(Protocol):
    """Tool-owned durable staging for marketplace handoff subjects."""

    def stage(
        self,
        record: MarketplaceQualifiedToolStage,
    ) -> MarketplaceQualifiedToolStageWriteResult:
        """Persist one immutable stage; conflict when identity collides with different payload."""
        ...

    def get(
        self,
        *,
        tenant_id: str,
        handoff_id: str,
    ) -> MarketplaceQualifiedToolStage | None:
        """Return staged record for tenant scope, or None when absent."""
        ...


__all__ = [
    "MarketplaceQualifiedToolStage",
    "MarketplaceQualifiedToolStageConflictError",
    "MarketplaceQualifiedToolStageError",
    "MarketplaceQualifiedToolStageIntegrityError",
    "MarketplaceQualifiedToolStageRepository",
    "MarketplaceQualifiedToolStageUnavailableError",
    "MarketplaceQualifiedToolStageWriteOutcome",
    "MarketplaceQualifiedToolStageWriteResult",
    "SCHEMA_MARKETPLACE_QUALIFIED_TOOL_STAGE_V1",
]
