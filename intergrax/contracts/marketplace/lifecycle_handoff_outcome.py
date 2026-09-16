# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed marketplace lifecycle handoff outcomes (ME-RB4)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_MARKETPLACE_LIFECYCLE_HANDOFF_OUTCOME_V1: Final = (
    "marketplace_lifecycle_handoff_outcome.v1"
)


class MarketplaceLifecycleHandoffStatus(StrEnum):
    """Handoff disposition — not installed/active/routable/executed."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    UNSUPPORTED = "unsupported"


class MarketplaceLifecycleHandoffReasonCode(StrEnum):
    HANDLER_MISSING = "handler_missing"
    UNSUPPORTED_CAPABILITY_KIND = "unsupported_capability_kind"
    INVALID_HANDOFF_REQUEST = "invalid_handoff_request"
    IDENTITY_MISMATCH = "identity_mismatch"
    DOMAIN_REJECTED = "domain_rejected"
    DOMAIN_UNAVAILABLE = "domain_unavailable"
    HANDLER_FAILED = "handler_failed"


class MarketplaceLifecycleHandoffOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_MARKETPLACE_LIFECYCLE_HANDOFF_OUTCOME_V1
    request_id: str = Field(min_length=1)
    status: MarketplaceLifecycleHandoffStatus
    domain_authority_id: str = Field(min_length=1)
    reason_code: MarketplaceLifecycleHandoffReasonCode | None = None
    reason_detail: str = ""
    domain_reference: str | None = None


__all__ = [
    "MarketplaceLifecycleHandoffOutcome",
    "MarketplaceLifecycleHandoffReasonCode",
    "MarketplaceLifecycleHandoffStatus",
    "SCHEMA_MARKETPLACE_LIFECYCLE_HANDOFF_OUTCOME_V1",
]
