# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.prompts.schema.prompt_schema import LocalizedContent


pytestmark = pytest.mark.unit


def _assert_basic(content: LocalizedContent, keyword: str) -> None:
    assert isinstance(content, LocalizedContent)
    assert isinstance(content.system, str)
    assert keyword.lower() in content.system.lower()


def test_websearch_serp_context_yaml() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    content = registry.resolve_localized(
        prompt_id="websearch_serp_context"
    )

    _assert_basic(content, "serp")


def test_websearch_grounded_context_yaml() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    content = registry.resolve_localized(
        prompt_id="websearch_grounded_context"
    )

    _assert_basic(content, "excerpts")


def test_websearch_chunk_rerank_notice_yaml() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    content = registry.resolve_localized(
        prompt_id="websearch_chunk_rerank_notice"
    )

    _assert_basic(content, "rerank")


def test_websearch_map_system_yaml() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    content = registry.resolve_localized(
        prompt_id="websearch_map_system"
    )

    _assert_basic(content, "page_excerpt")


def test_websearch_reduce_system_yaml() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    content = registry.resolve_localized(
        prompt_id="websearch_reduce_system"
    )

    _assert_basic(content, "fact_cards")


def test_websearch_mapreduce_no_evidence_yaml() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    content = registry.resolve_localized(
        prompt_id="websearch_mapreduce_no_evidence"
    )

    _assert_basic(content, "no answer")


def test_websearch_mapreduce_final_header_yaml() -> None:
    registry = YamlPromptRegistry.create_default(load=True)

    content = registry.resolve_localized(
        prompt_id="websearch_mapreduce_final_header"
    )

    _assert_basic(content, "synthesized")
