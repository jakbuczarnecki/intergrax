# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Skill domain marketplace lifecycle handoff boundary (ME-RB4-C1)."""

from __future__ import annotations

from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.lifecycle_handoff.ack import DomainLifecycleHandoffAck

SCHEMA_SKILL_LIFECYCLE_HANDOFF_PAYLOAD_V1: Final = "skill_lifecycle_handoff_payload.v1"

SKILL_DOMAIN_AUTHORITY_ID = "skill_domain"


class SkillLifecycleHandoffError(Exception):
    """Base error for Skill domain marketplace lifecycle handoff."""


class SkillLifecycleHandoffUnavailableError(SkillLifecycleHandoffError):
    """Skill lifecycle authority cannot accept handoff (transient or dependency down)."""


class SkillLifecycleHandoffPayload(BaseModel):
    """Marketplace→Skill handoff slice; composition semantics remain Skill-owned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_SKILL_LIFECYCLE_HANDOFF_PAYLOAD_V1
    operation_id: str = Field(min_length=1)
    host_profile_id: str = Field(min_length=1)
    capability_identity_key: CapabilityIdentityKey


class SkillMarketplaceLifecycleHandoffPort(Protocol):
    """Skill-owned lifecycle handoff boundary consumed by marketplace adapters."""

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: SkillLifecycleHandoffPayload,
        *,
        request_id: str,
        correlation_id: str | None,
    ) -> DomainLifecycleHandoffAck: ...


__all__ = [
    "SCHEMA_SKILL_LIFECYCLE_HANDOFF_PAYLOAD_V1",
    "SKILL_DOMAIN_AUTHORITY_ID",
    "SkillLifecycleHandoffError",
    "SkillLifecycleHandoffPayload",
    "SkillLifecycleHandoffUnavailableError",
    "SkillMarketplaceLifecycleHandoffPort",
]
