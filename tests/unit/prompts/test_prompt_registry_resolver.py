# © Artur Czarnecki. All rights reserved.

"""PE-4: shared prompt registry resolver."""

from __future__ import annotations

import pytest

from intergrax.prompts.registry.prompt_registry_resolver import (
    DEFAULT_PROMPT_CATALOG,
    resolve_yaml_prompt_registry,
)
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resolve_yaml_prompt_registry_returns_injected_instance() -> None:
    existing = resolve_yaml_prompt_registry(catalog_path=DEFAULT_PROMPT_CATALOG)
    resolved = resolve_yaml_prompt_registry(registry=existing)
    assert resolved is existing


def test_resolve_yaml_prompt_registry_uses_catalog_path() -> None:
    registry = resolve_yaml_prompt_registry(catalog_path="prompts")
    assert isinstance(registry, YamlPromptRegistry)
