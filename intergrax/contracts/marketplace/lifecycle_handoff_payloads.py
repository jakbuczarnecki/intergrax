# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace lifecycle handoff payload aliases — canonical models are domain-owned (ME-RB4-C1)."""

from __future__ import annotations

from intergrax.contracts.agent_distribution.marketplace_lifecycle_handoff import (
    SCHEMA_AGENT_LIFECYCLE_HANDOFF_PAYLOAD_V1,
    AgentLifecycleHandoffPayload,
)
from intergrax.contracts.skills.marketplace_lifecycle_handoff import (
    SCHEMA_SKILL_LIFECYCLE_HANDOFF_PAYLOAD_V1,
    SkillLifecycleHandoffPayload,
)
from intergrax.contracts.tools.marketplace_lifecycle_handoff import (
    SCHEMA_TOOL_LIFECYCLE_HANDOFF_PAYLOAD_V1,
    ToolLifecycleHandoffPayload,
)

__all__ = [
    "AgentLifecycleHandoffPayload",
    "SCHEMA_AGENT_LIFECYCLE_HANDOFF_PAYLOAD_V1",
    "SCHEMA_SKILL_LIFECYCLE_HANDOFF_PAYLOAD_V1",
    "SCHEMA_TOOL_LIFECYCLE_HANDOFF_PAYLOAD_V1",
    "SkillLifecycleHandoffPayload",
    "ToolLifecycleHandoffPayload",
]
