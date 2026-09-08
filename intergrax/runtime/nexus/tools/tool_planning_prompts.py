# © Artur Czarnecki. All rights reserved.

"""Planner prompt resolution for Tier-1 tool planning (Phase U-Typ.2, PE-4)."""

from __future__ import annotations

from intergrax.prompts.registry.prompt_registry_resolver import resolve_yaml_prompt_registry
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry

GENERIC_INVESTIGATION_POLICY_PROMPT_ID = "tools_investigation_policy"


def _resolve_registry(
    *,
    registry: YamlPromptRegistry | None = None,
    catalog_path: str | None = None,
) -> YamlPromptRegistry:
    return resolve_yaml_prompt_registry(registry=registry, catalog_path=catalog_path)


def planner_prompt(
    *,
    prompt_id: str = "tools_agent_planner",
    registry: YamlPromptRegistry | None = None,
    catalog_path: str | None = None,
) -> str:
    reg = _resolve_registry(registry=registry, catalog_path=catalog_path)
    return reg.resolve_localized(prompt_id).system


def investigation_policy_prompt(
    *,
    prompt_id: str = GENERIC_INVESTIGATION_POLICY_PROMPT_ID,
    registry: YamlPromptRegistry | None = None,
    catalog_path: str | None = None,
) -> str:
    reg = _resolve_registry(registry=registry, catalog_path=catalog_path)
    return reg.resolve_localized(prompt_id).system


def composed_investigation_policy_prompt(
    *,
    overlay_prompt_id: str,
    registry: YamlPromptRegistry | None = None,
    catalog_path: str | None = None,
) -> str:
    """Generic atomic transport policy plus scenario-specific investigation overlay."""
    prompt_kwargs = {"registry": registry, "catalog_path": catalog_path}
    generic = investigation_policy_prompt(**prompt_kwargs)
    overlay = investigation_policy_prompt(prompt_id=overlay_prompt_id, **prompt_kwargs)
    return f"{generic.strip()}\n\n{overlay.strip()}"


def system_prompt(
    *,
    registry: YamlPromptRegistry | None = None,
    catalog_path: str | None = None,
) -> str:
    reg = _resolve_registry(registry=registry, catalog_path=catalog_path)
    return reg.resolve_localized("tools_agent_system").system


def system_context_template(
    *,
    registry: YamlPromptRegistry | None = None,
    catalog_path: str | None = None,
) -> str:
    """
    Legacy-compatible template containing ``{context}`` placeholder.
    Formatting is done later via ``.format(context=...)``.
    """
    reg = _resolve_registry(registry=registry, catalog_path=catalog_path)
    localized = reg.resolve_localized("tools_agent_context")
    return localized.user_template or ""
