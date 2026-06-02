# © Artur Czarnecki. All rights reserved.

"""Planner prompt resolution for Tier-1 tool planning (Phase U-Typ.2)."""

from __future__ import annotations

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry


def planner_prompt() -> str:
    registry = YamlPromptRegistry.create_default(load=True)
    return registry.resolve_localized("tools_agent_planner").system


def system_prompt() -> str:
    registry = YamlPromptRegistry.create_default(load=True)
    return registry.resolve_localized("tools_agent_system").system


def system_context_template() -> str:
    """
    Legacy-compatible template containing ``{context}`` placeholder.
    Formatting is done later via ``.format(context=...)``.
    """
    registry = YamlPromptRegistry.create_default(load=True)
    localized = registry.resolve_localized("tools_agent_context")
    return localized.user_template or ""
