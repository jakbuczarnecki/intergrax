# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Builtin first-party agent capability metadata — declarative, non-executable."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.agent_distribution.agent_capability_metadata import (
    AgentCapabilityDescriptor,
    AgentCapabilityMetadataProvider,
)

# Declared contract metadata mirrored from Tier-2 agent packages without importing agents.
_BUILTIN_AGENT_CAPABILITY_DESCRIPTORS: tuple[AgentCapabilityDescriptor, ...] = (
    AgentCapabilityDescriptor(
        contract_id="echo",
        agent_version="1.0.0",
        capabilities=("echo.basic",),
        skill_ids=("harness.tool_smoke",),
        tool_ids=(),
    ),
    AgentCapabilityDescriptor(
        contract_id="legal",
        agent_version="0.1.0",
        capabilities=("legal.review",),
        skill_ids=("legal.contract_review",),
        tool_ids=(),
    ),
    AgentCapabilityDescriptor(
        contract_id="research",
        agent_version="0.1.0",
        capabilities=("research.web_search", "research.pipeline"),
        skill_ids=("research.literature_scan",),
        tool_ids=(),
    ),
    AgentCapabilityDescriptor(
        contract_id="research-summary",
        agent_version="0.1.0",
        capabilities=("research.summarize",),
        skill_ids=(),
        tool_ids=(),
    ),
)


class BuiltinAgentCapabilityMetadataProvider:
    """Platform-safe builtin metadata adapter for harness/reference agents."""

    def list_agent_capability_descriptors(self) -> Sequence[AgentCapabilityDescriptor]:
        return _BUILTIN_AGENT_CAPABILITY_DESCRIPTORS


def default_agent_capability_metadata_provider() -> AgentCapabilityMetadataProvider:
    """Return the platform builtin agent capability metadata provider."""
    return BuiltinAgentCapabilityMetadataProvider()
