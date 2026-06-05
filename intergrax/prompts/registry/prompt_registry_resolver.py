# © Artur Czarnecki. All rights reserved.

"""Resolve ``YamlPromptRegistry`` from injected instance or catalog path (Phase PE-4)."""

from __future__ import annotations

from pathlib import Path

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry

DEFAULT_PROMPT_CATALOG = Path("prompts")


def resolve_yaml_prompt_registry(
    *,
    registry: YamlPromptRegistry | None = None,
    catalog_path: str | Path | None = None,
    load: bool = True,
) -> YamlPromptRegistry:
    """
    Return an existing registry or materialize one from ``catalog_path``.

    Precedence: explicit ``registry`` → ``catalog_path`` → default ``prompts/`` catalog.
    """
    if registry is not None:
        return registry
    path = str(catalog_path) if catalog_path is not None else None
    return YamlPromptRegistry.create_default(path=path, load=load)
