# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed vertical payloads for marketplace lifecycle handoff (ME-RB4)."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey

SCHEMA_AGENT_LIFECYCLE_HANDOFF_PAYLOAD_V1: Final = "agent_lifecycle_handoff_payload.v1"
SCHEMA_TOOL_LIFECYCLE_HANDOFF_PAYLOAD_V1: Final = "tool_lifecycle_handoff_payload.v1"
SCHEMA_SKILL_LIFECYCLE_HANDOFF_PAYLOAD_V1: Final = "skill_lifecycle_handoff_payload.v1"


class AgentLifecycleHandoffPayload(BaseModel):
    """Marketplace-safe agent handoff slice — full AC-3/AC-4 request built by domain authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_LIFECYCLE_HANDOFF_PAYLOAD_V1
    operation_id: str = Field(min_length=1)
    application_id: str = Field(min_length=1)
    application_environment_id: str = Field(min_length=1)
    catalog_entry_id: str = Field(min_length=1)
    capability_identity_key: CapabilityIdentityKey


class ToolLifecycleHandoffPayload(BaseModel):
    """Tool domain lifecycle handoff — host/profile scope; domain owns registration semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TOOL_LIFECYCLE_HANDOFF_PAYLOAD_V1
    operation_id: str = Field(min_length=1)
    host_profile_id: str = Field(min_length=1)
    capability_identity_key: CapabilityIdentityKey


class SkillLifecycleHandoffPayload(BaseModel):
    """Skill domain lifecycle handoff — host/profile scope; domain owns resolve/composition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_SKILL_LIFECYCLE_HANDOFF_PAYLOAD_V1
    operation_id: str = Field(min_length=1)
    host_profile_id: str = Field(min_length=1)
    capability_identity_key: CapabilityIdentityKey


__all__ = [
    "AgentLifecycleHandoffPayload",
    "SCHEMA_AGENT_LIFECYCLE_HANDOFF_PAYLOAD_V1",
    "SCHEMA_SKILL_LIFECYCLE_HANDOFF_PAYLOAD_V1",
    "SCHEMA_TOOL_LIFECYCLE_HANDOFF_PAYLOAD_V1",
    "SkillLifecycleHandoffPayload",
    "ToolLifecycleHandoffPayload",
]
