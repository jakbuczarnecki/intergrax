# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.agent_distribution.agent_capability_metadata import (
    AgentCapabilityDescriptor,
    AgentCapabilityDescriptorConflictError,
    merge_agent_capability_descriptors,
)


def test_agent_capability_descriptor_rejects_empty_contract_id() -> None:
    with pytest.raises(ValueError):
        AgentCapabilityDescriptor(contract_id=" ", agent_version="1.0.0")


def test_merge_agent_capability_descriptors_deduplicates_identical_rows() -> None:
    descriptor = AgentCapabilityDescriptor(
        contract_id="agent-a",
        agent_version="1.0.0",
        capabilities=("knowledge.search",),
        skill_ids=("research",),
        tool_ids=("rag.retrieve",),
    )
    merged = merge_agent_capability_descriptors([descriptor, descriptor])
    assert merged == (descriptor,)


def test_merge_agent_capability_descriptors_raises_on_conflict() -> None:
    left = AgentCapabilityDescriptor(contract_id="agent-a", agent_version="1.0.0")
    right = AgentCapabilityDescriptor(contract_id="agent-a", agent_version="2.0.0")
    with pytest.raises(AgentCapabilityDescriptorConflictError, match="conflicting"):
        merge_agent_capability_descriptors([left, right])
