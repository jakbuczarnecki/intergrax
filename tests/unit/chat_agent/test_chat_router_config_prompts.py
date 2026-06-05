# © Artur Czarnecki. All rights reserved.

"""Chat router YAML prompt assets (no legacy chat_router module)."""

from __future__ import annotations

import pytest

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry

pytestmark = pytest.mark.unit


def test_chat_router_general_and_tool_prompts_load_from_yaml() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    general = (registry.resolve_localized("chat_router_general").system or "").strip()
    tools = (registry.resolve_localized("chat_router_tool").system or "").strip()

    assert general
    assert tools
