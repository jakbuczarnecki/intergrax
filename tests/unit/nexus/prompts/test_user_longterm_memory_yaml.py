# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.prompts.schema.prompt_schema import LocalizedContent, LocalizedPromptDocument



pytestmark = pytest.mark.unit


def test_user_longterm_memory_yaml_can_be_loaded(tmp_path: Path) -> None:
    """
    Contract test ensuring that:
    - user_longterm_memory/1.yaml exists,
    - can be loaded by YamlPromptRegistry,
    - id and version are taken from file content (not filename),
    - English locale contains system section.
    """

    # Arrange
    registry = YamlPromptRegistry.create_default(load=True)

    # Act
    content = registry.resolve_localized(
        prompt_id="user_longterm_memory"
    )

    # Assert – type contract
    assert isinstance(content, LocalizedContent)

    # Assert – structural contract
    assert hasattr(content, "system")
    assert isinstance(content.system, str)

    # Minimal semantic check
    assert "USER LONG-TERM MEMORY" in content.system
