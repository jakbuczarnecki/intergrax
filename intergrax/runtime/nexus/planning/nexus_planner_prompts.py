# © Artur Czarnecki. All rights reserved.

"""Nexus planner prompt resolution (COG-2.1)."""

from __future__ import annotations

from intergrax.prompts.registry.prompt_registry_resolver import resolve_yaml_prompt_registry
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry


def nexus_task_planner_prompt(
    *,
    registry: YamlPromptRegistry | None = None,
    catalog_path: str | None = None,
    prompt_id: str = "nexus_task_planner",
    agent_ids: list[str],
    task_message: str,
    capability: str,
    classification: str,
) -> str:
    reg = resolve_yaml_prompt_registry(registry=registry, catalog_path=catalog_path)
    localized = reg.resolve_localized(prompt_id)
    system = localized.system or ""
    user_template = localized.user_template
    if user_template:
        user_body = user_template.format(
            agent_ids=agent_ids,
            task_message=task_message,
            capability=capability,
            classification=classification,
        )
        return f"{system}\n\n{user_body}".strip()
    return (
        f"{system}\n"
        f"Agent ids: {agent_ids}\n"
        f"Task message: {task_message!r}\n"
        f"Capability: {capability}\n"
        f"Classification: {classification}"
    )
