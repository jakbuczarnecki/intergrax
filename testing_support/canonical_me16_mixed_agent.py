# © Artur Czarnecki. All rights reserved.

"""Deterministic ME-16 mixed-capability worker identity and output helpers."""

from __future__ import annotations

from typing import Final

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
    "ME14_TOOL_LOGICAL_ID",
    "ME15_SKILL_LOGICAL_ID",
    "expected_mixed_output_for_releases",
]
