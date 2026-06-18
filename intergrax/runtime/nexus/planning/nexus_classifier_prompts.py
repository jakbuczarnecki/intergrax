# © Artur Czarnecki. All rights reserved.

"""Nexus classifier prompt resolution (COG-LC-S6)."""

from __future__ import annotations

from intergrax.prompts.registry.prompt_registry_resolver import resolve_yaml_prompt_registry
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry


def nexus_task_classifier_prompt(
    *,
    registry: YamlPromptRegistry | None = None,
    catalog_path: str | None = None,
    prompt_id: str = "nexus_task_classifier",
    capabilities: tuple[str, ...],
    task_message: str,
) -> str:
    reg = resolve_yaml_prompt_registry(registry=registry, catalog_path=catalog_path)
    localized = reg.resolve_localized(prompt_id)
    system = localized.system or ""
    user_template = localized.user_template
    options = ", ".join(repr(cap) for cap in capabilities)
    if user_template:
        user_body = user_template.format(
            capabilities=options,
            task_message=task_message,
        )
        return f"{system}\n\n{user_body}".strip()
    return f"{system}\nCapabilities: {options}\nUser message: {task_message!r}"
