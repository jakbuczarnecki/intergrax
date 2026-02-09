# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.prompts.history_prompt_builder import (
    DefaultHistorySummaryPromptBuilder,
    HistorySummaryPromptBundle,
)
from intergrax.runtime.nexus.responses.response_schema import (
    RuntimeRequest,
    HistoryCompressionStrategy,
)


pytestmark = pytest.mark.unit


def test_history_summary_prompt_builder_uses_yaml_registry() -> None:
    """
    Contract test for DefaultHistorySummaryPromptBuilder:

    - system prompt is loaded from YAML Prompt Registry
    - builder returns HistorySummaryPromptBundle
    """

    # Arrange    
    registry = YamlPromptRegistry.create_default(load=True)
    
    registry.load_all()

    builder = DefaultHistorySummaryPromptBuilder(
        config=None,  # not used by builder
        prompt_registry=registry,
    )

    # Act
    bundle = builder.build_history_summary_prompt(
        request=RuntimeRequest(
            message="",
            session_id="test_session_1",
            user_id="test_user_1"),
        strategy=HistoryCompressionStrategy.OFF,
        older_messages=[],
        tail_messages=[],
    )

    # Assert
    assert isinstance(bundle, HistorySummaryPromptBundle)
    assert isinstance(bundle.system_prompt, str)
    assert bundle.system_prompt.strip() != ""

    # Minimal semantic check (ties test to YAML, not hardcoded text)
    assert "history" in bundle.system_prompt.lower()
