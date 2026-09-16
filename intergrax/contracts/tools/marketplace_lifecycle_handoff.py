# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool domain marketplace lifecycle handoff boundary (ME-RB4-C1)."""

from __future__ import annotations

from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.lifecycle_handoff.ack import DomainLifecycleHandoffAck

SCHEMA_TOOL_LIFECYCLE_HANDOFF_PAYLOAD_V1: Final = "tool_lifecycle_handoff_payload.v1"

TOOL_DOMAIN_AUTHORITY_ID = "tool_domain"


class ToolLifecycleHandoffError(Exception):
    """Base error for Tool domain marketplace lifecycle handoff."""


class ToolLifecycleHandoffUnavailableError(ToolLifecycleHandoffError):
    """Tool lifecycle authority cannot accept handoff (transient or dependency down)."""


class ToolLifecycleHandoffPayload(BaseModel):
    """Marketplace→Tool handoff slice; registration semantics remain Tool-owned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TOOL_LIFECYCLE_HANDOFF_PAYLOAD_V1
    operation_id: str = Field(min_length=1)
    host_profile_id: str = Field(min_length=1)
    capability_identity_key: CapabilityIdentityKey


class ToolMarketplaceLifecycleHandoffPort(Protocol):
    """Tool-owned lifecycle handoff boundary consumed by marketplace adapters."""

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: ToolLifecycleHandoffPayload,
        *,
        request_id: str,
        correlation_id: str | None,
    ) -> DomainLifecycleHandoffAck: ...


__all__ = [
    "SCHEMA_TOOL_LIFECYCLE_HANDOFF_PAYLOAD_V1",
    "TOOL_DOMAIN_AUTHORITY_ID",
    "ToolLifecycleHandoffError",
    "ToolLifecycleHandoffPayload",
    "ToolLifecycleHandoffUnavailableError",
    "ToolMarketplaceLifecycleHandoffPort",
]
