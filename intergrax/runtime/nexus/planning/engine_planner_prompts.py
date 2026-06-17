# © Artur Czarnecki. All rights reserved.

"""Agent-level engine planner prompt resolution (COG-2.3 / COG-LC-S2)."""

from __future__ import annotations

from intergrax.prompts.registry.prompt_registry_resolver import resolve_yaml_prompt_registry
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry


def resolve_engine_planner_system_prompt(
    *,
    prompt_id: str,
    registry: YamlPromptRegistry | None = None,
    catalog_path: str | None = None,
) -> str:
    """Resolve registry-backed system prompt for Plane 2 engine step cognition."""
    reg = resolve_yaml_prompt_registry(registry=registry, catalog_path=catalog_path)
    localized = reg.resolve_localized(prompt_id)
    return localized.system or ""
