# © Artur Czarnecki. All rights reserved.

"""Reusable CapabilityBundle presets for Tier-3 reference hosts (APP-EVOL-8.5)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile.bundles import CapabilityBundle
from intergrax.applications.contracts.environment_profile.presets import (
    harness_lab_capability_bundle,
    harness_memory_profile,
    lab_reference_tool_profile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import MemoryProfile
from intergrax.tools.contracts.tool_profile import ToolProfile

__all__ = [
    "CapabilityBundle",
    "MemoryProfile",
    "ToolProfile",
    "harness_lab_capability_bundle",
    "harness_memory_profile",
    "harness_platform_tool_profile",
    "lab_reference_tool_profile",
]


def harness_platform_tool_profile() -> ToolProfile:
    """Tool availability for harness-only platform skill hosts (pairs with harness skill bundle)."""
    return lab_reference_tool_profile(harness_tools=False)
