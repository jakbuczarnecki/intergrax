# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.chains.langchain_qa_chain import _default_prompt_builder
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry


def test_langchain_qa_yaml_contract() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    content = registry.resolve_localized(prompt_id="langchain_qa")

    assert isinstance(content.system, str)
    assert isinstance(content.user_template, str)

    assert "{{context}}" in content.user_template
    assert "{{question}}" in content.user_template


def test_langchain_qa_prompt_rendering() -> None:
    result = _default_prompt_builder(
        context="Paris is the capital of France.",
        question="What is the capital of France?"
    )

    assert "Paris is the capital of France." in result
    assert "What is the capital of France?" in result
    assert "FINAL ANSWER" in result
