# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.rag.answers.builders.default_prompts import default_rag_system_instruction
from intergrax.rag.answers.builders.prompt_builder import DefaultPromptBuilder


pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _assert_non_empty_str(value: str) -> None:
    assert isinstance(value, str)
    assert value.strip()


# ----------------------------------------------------------------------
# Core contract
# ----------------------------------------------------------------------

def test_prompt_builder_builds_prompt_with_context_and_question() -> None:

    builder = DefaultPromptBuilder()

    prompt = builder.build(
        query="What is AI?",
        context="Artificial Intelligence is the simulation of human intelligence."
    )

    _assert_non_empty_str(prompt)

    assert "What is AI?" in prompt
    assert "Artificial Intelligence is the simulation" in prompt
    assert "Answer:" in prompt


# ----------------------------------------------------------------------
# Override behavior
# ----------------------------------------------------------------------

def test_prompt_builder_uses_custom_system_prompt_when_provided() -> None:

    builder = DefaultPromptBuilder(
        system_prompt="CUSTOM_SYSTEM_PROMPT"
    )

    prompt = builder.build(
        query="Question",
        context="Context"
    )

    assert prompt.startswith("CUSTOM_SYSTEM_PROMPT")
    assert "Context:" in prompt
    assert "Question:" in prompt


# ----------------------------------------------------------------------
# YAML registry integration
# ----------------------------------------------------------------------

def test_default_rag_system_prompt_is_available() -> None:

    prompt = default_rag_system_instruction()

    assert isinstance(prompt, str)
    assert prompt.strip()