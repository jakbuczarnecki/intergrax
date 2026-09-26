# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool-owned marketplace qualified stage context contracts (S24-GAP-02-P2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text

SCHEMA_MARKETPLACE_QUALIFIED_TOOL_STAGE_CONTEXT_V1: Final = (
    "marketplace_qualified_tool_stage_context.v1"
)


class MarketplaceQualifiedToolStageContext(BaseModel):
    """Immutable association between tenant-distinct handoff and acquisition identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["marketplace_qualified_tool_stage_context.v1"] = (
        SCHEMA_MARKETPLACE_QUALIFIED_TOOL_STAGE_CONTEXT_V1
    )
    handoff_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    acquisition_request_id: str = Field(min_length=1)

    @field_validator("handoff_id", "tenant_id", "acquisition_request_id")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        return require_non_empty_text(value, label="identity")


class MarketplaceQualifiedToolStageContextAssociationWriteOutcome(StrEnum):
    CREATED = "created"
    ALREADY_RECORDED_IDENTICAL = "already_recorded_identical"


class MarketplaceQualifiedToolStageContextAssociationWriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: MarketplaceQualifiedToolStageContextAssociationWriteOutcome


class MarketplaceQualifiedToolStageContextError(Exception):
    """Base error for Tool marketplace qualified stage context."""


class MarketplaceQualifiedToolStageContextAssociationConflictError(
    MarketplaceQualifiedToolStageContextError,
):
    """Same handoff_id recorded with a different immutable association payload."""


class MarketplaceQualifiedToolStageContextAssociationUnavailableError(
    MarketplaceQualifiedToolStageContextError,
):
    """Association backend or dependency is unavailable."""


class MarketplaceQualifiedToolStageContextAssociationIntegrityError(
    MarketplaceQualifiedToolStageContextError,
):
    """Persistent association cannot be reconstructed into the canonical context."""


class MarketplaceQualifiedToolStageContextResolverError(
    MarketplaceQualifiedToolStageContextError,
):
    """Base resolver failure."""


class MarketplaceQualifiedToolStageContextNotFoundError(
    MarketplaceQualifiedToolStageContextResolverError,
):
    """No association exists for the parsed handoff identity."""


class MarketplaceQualifiedToolStageContextResolverConflictError(
    MarketplaceQualifiedToolStageContextResolverError,
):
    """Resolver inputs disagree with stored association facts."""


class MarketplaceQualifiedToolStageContextResolverIntegrityError(
    MarketplaceQualifiedToolStageContextResolverError,
):
    """Stored association or handoff identity failed integrity validation."""


class MarketplaceQualifiedToolStageContextResolverUnavailableError(
    MarketplaceQualifiedToolStageContextResolverError,
):
    """Resolver dependency is unavailable."""


class MarketplaceQualifiedToolStageContextResolverNotSupportedError(
    MarketplaceQualifiedToolStageContextResolverError,
):
    """Strategy or reference shape is not supported by this resolver."""


@runtime_checkable
class MarketplaceQualifiedToolStageContextAssociationRepository(Protocol):
    """Durable handoff→tenant context association (handoff_id is globally distinct)."""

    def record(
        self,
        association: MarketplaceQualifiedToolStageContext,
    ) -> MarketplaceQualifiedToolStageContextAssociationWriteResult:
        """Persist one immutable association; conflict when identity collides."""
        ...

    def get_by_handoff_id(
        self,
        handoff_id: str,
    ) -> MarketplaceQualifiedToolStageContext | None:
        """Return association for handoff_id, or None when absent."""
        ...


@runtime_checkable
class MarketplaceQualifiedToolStageContextResolver(Protocol):
    """Resolve tenant-safe context for Marketplace Tool qualification/binding."""

    def resolve_for_qualification(
        self,
        *,
        acquisition_request_id: str,
        domain_handoff_reference: str,
        strategy_id: str,
    ) -> MarketplaceQualifiedToolStageContext:
        """Resolve context from stable typed facts — no UCA implementation objects."""
        ...


__all__ = [
    "MarketplaceQualifiedToolStageContext",
    "MarketplaceQualifiedToolStageContextAssociationConflictError",
    "MarketplaceQualifiedToolStageContextAssociationIntegrityError",
    "MarketplaceQualifiedToolStageContextAssociationRepository",
    "MarketplaceQualifiedToolStageContextAssociationUnavailableError",
    "MarketplaceQualifiedToolStageContextAssociationWriteOutcome",
    "MarketplaceQualifiedToolStageContextAssociationWriteResult",
    "MarketplaceQualifiedToolStageContextError",
    "MarketplaceQualifiedToolStageContextNotFoundError",
    "MarketplaceQualifiedToolStageContextResolver",
    "MarketplaceQualifiedToolStageContextResolverConflictError",
    "MarketplaceQualifiedToolStageContextResolverError",
    "MarketplaceQualifiedToolStageContextResolverIntegrityError",
    "MarketplaceQualifiedToolStageContextResolverNotSupportedError",
    "MarketplaceQualifiedToolStageContextResolverUnavailableError",
    "SCHEMA_MARKETPLACE_QUALIFIED_TOOL_STAGE_CONTEXT_V1",
]
