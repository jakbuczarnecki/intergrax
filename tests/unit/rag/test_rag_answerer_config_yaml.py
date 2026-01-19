# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.rag.rag_answerer import AnswererConfig
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry


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

def test_answerer_config_ensure_prompts_loads_all_from_yaml() -> None:
    cfg = AnswererConfig()
    cfg.ensure_prompts()

    _assert_non_empty_str(cfg.system_instructions)

    assert "{context}" in cfg.system_context_template
    assert "{question}" in cfg.user_question_template
    assert "{instruction}" in cfg.user_instruction_template

    assert "{answer}" in cfg.summary_prompt_template
    _assert_non_empty_str(cfg.summary_system_instruction)


# ----------------------------------------------------------------------
# Override behavior
# ----------------------------------------------------------------------

def test_answerer_config_ensure_prompts_does_not_override_explicit_values() -> None:
    cfg = AnswererConfig(
        system_instructions="CUSTOM_SYSTEM",
        system_context_template="CUSTOM_CONTEXT {context}",
    )

    cfg.ensure_prompts()

    # Explicit values must stay untouched
    assert cfg.system_instructions == "CUSTOM_SYSTEM"
    assert cfg.system_context_template == "CUSTOM_CONTEXT {context}"

    # Others should be filled from YAML
    assert cfg.user_question_template
    assert cfg.user_instruction_template
    assert cfg.summary_prompt_template
    assert cfg.summary_system_instruction


# ----------------------------------------------------------------------
# Registry integration
# ----------------------------------------------------------------------

def test_answerer_config_yaml_registry_contains_all_required_prompts() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    assert registry.resolve_localized("rag_answerer_system")
    assert registry.resolve_localized("rag_answerer_context")
    assert registry.resolve_localized("rag_answerer_user_question")
    assert registry.resolve_localized("rag_answerer_user_instruction")
    assert registry.resolve_localized("rag_answerer_summary_prompt")
    assert registry.resolve_localized("rag_answerer_summary_system")
