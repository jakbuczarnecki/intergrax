# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AHI craft trigger — catalog miss and budget gate (ECC-6)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.codecraft.profile import CodeCraftProfile


@dataclass(frozen=True)
class CraftAdaptiveDecision:
    suggest_craft: bool
    auto_invoke: bool
    reason: str


def evaluate_craft_trigger(
    *,
    requested_tool_id: str,
    catalog_has_tool: bool,
    profile: CodeCraftProfile | None,
    budget_exhausted: bool = False,
    adaptive_enabled: bool = False,
) -> CraftAdaptiveDecision:
    """Decide whether L4 should suggest or auto-invoke codecraft."""
    if budget_exhausted:
        return CraftAdaptiveDecision(False, False, "budget_exhausted")
    if profile is None or not profile.generation_allowed():
        return CraftAdaptiveDecision(False, False, "codecraft_disabled")
    if catalog_has_tool:
        return CraftAdaptiveDecision(False, False, "catalog_tool_available")
    if not adaptive_enabled:
        return CraftAdaptiveDecision(True, False, f"catalog_miss:{requested_tool_id}")
    if profile.mode == "autonomous":
        return CraftAdaptiveDecision(True, True, f"catalog_miss_auto:{requested_tool_id}")
    return CraftAdaptiveDecision(True, False, f"catalog_miss_supervised:{requested_tool_id}")
