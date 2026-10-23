# © Artur Czarnecki. All rights reserved.

"""Deterministic ME-16 mixed-capability worker identity and output helpers."""

from __future__ import annotations

from typing import Final

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.capability_catalog import CapabilityReleaseIdentity
from testing_support.canonical_me14_echo_tool import (
    ME14_TOOL_LOGICAL_ID,
    expected_output_for_release as expected_tool_output_for_release,
)
from testing_support.canonical_me15_reference_skill import (
    ME15_SKILL_LOGICAL_ID,
    instruction_marker_for_release,
)

ME16_MIXED_CONTRACT_ID: Final = "me16-mixed-capability-agent"
ME16_MIXED_CAPABILITY: Final = "me16.mixed.capability"
ME16_MARKETPLACE_LOGICAL_ID: Final = "agents.me16.mixed-worker"
ME16_LISTING_ID: Final = "listing-me16-mixed-worker"
ME16_MIXED_TENANT: Final = "tenant-me16"
ME16_MIXED_TASK_INPUT: Final = "run-mixed-capability"
ME16_MIXED_CONTRACT_VERSION: Final = "1.0.0"
ME16_MIXED_TOOL_ID: Final = "tools.me14.canonical-echo"


def me16_mixed_reference_contract() -> AgentContract:
    """Canonical ME-16 contract snapshot for digest-bound install authority (proofs)."""
    return AgentContract(
        id=ME16_MIXED_CONTRACT_ID,
        name="ME-16 Mixed Capability Worker",
        description="Consumes bound skill context and invokes catalog tool.",
        version=ME16_MIXED_CONTRACT_VERSION,
        capabilities=[ME16_MIXED_CAPABILITY, ME16_MIXED_TOOL_ID],
        skills=[],
        extra_tools=[],
        allowed_tools=[ME16_MIXED_TOOL_ID],
        risk_level=AgentRiskLevel.LOW,
        lifecycle_state=AgentLifecycleState.PRODUCTION,
        production_eligible=True,
        owner_team="platform",
        owner_contact="harness@intergrax",
        on_call_contact="harness@intergrax",
        runbook_ref="docs/project/architecture/CAPABILITY_MARKETPLACE_ENGINE.md",
        modality_profile_id="me16.mixed",
        input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        validation_rules=["structured_output"],
        failure_modes=["tool_invoke_failed", "skill_binding_missing"],
        max_steps=1,
    )


def expected_mixed_output_for_releases(
    *,
    skill_release: CapabilityReleaseIdentity,
    tool_release: CapabilityReleaseIdentity,
) -> str:
    marker = instruction_marker_for_release(skill_release)
    tool_part = expected_tool_output_for_release(tool_release)
    return f"{marker}|{tool_part}"


__all__ = [
    "ME16_LISTING_ID",
    "ME16_MARKETPLACE_LOGICAL_ID",
    "ME16_MIXED_CAPABILITY",
    "ME16_MIXED_CONTRACT_ID",
    "ME16_MIXED_TASK_INPUT",
    "ME16_MIXED_TENANT",
    "ME16_MIXED_CONTRACT_VERSION",
    "ME16_MIXED_TOOL_ID",
    "ME14_TOOL_LOGICAL_ID",
    "ME15_SKILL_LOGICAL_ID",
    "expected_mixed_output_for_releases",
    "me16_mixed_reference_contract",
]
