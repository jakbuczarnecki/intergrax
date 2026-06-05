# © Artur Czarnecki. All rights reserved.

"""Tier-3 prompt registry wiring (Phase PE-2)."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications.contracts.environment_profile import PromptProfile
from intergrax.prompts.registry.prompt_registry_protocol import PromptRegistryProtocol
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry

DEFAULT_PROMPT_CATALOG = Path("prompts")


def resolve_prompt_catalog_path(profile: PromptProfile) -> Path:
    """Resolve effective YAML catalog directory."""
    if profile.catalog_path is not None:
        return profile.catalog_path
    return DEFAULT_PROMPT_CATALOG


def resolve_prompt_registry(profile: PromptProfile) -> YamlPromptRegistry:
    """Materialize ``YamlPromptRegistry`` from Tier-3 ``PromptProfile``."""
    catalog_path = resolve_prompt_catalog_path(profile)
    return YamlPromptRegistry.create_default(
        path=str(catalog_path),
        load=profile.load_on_startup,
    )


def resolve_prompt_registry_protocol(profile: PromptProfile) -> PromptRegistryProtocol:
    """Return registry as :class:`PromptRegistryProtocol` for builder injection."""
    return resolve_prompt_registry(profile)
