# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy
from intergrax.rag.document_splitters.engine.chunking_engine import ChunkingEngine
from intergrax.rag.document_splitters.registry.plugin_registry import (
    apply_chunking_strategy_plugins,
    list_chunking_strategy_plugins,
    register_chunking_strategy_plugin,
)
from intergrax.rag.document_splitters.registry.strategy_registry import ChunkingStrategyRegistry
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.profiles.rag_profile_validator import assert_rag_profile_wiring

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _PassthroughPluginStrategy(BaseChunkingStrategy):
    @classmethod
    def strategy_id(cls) -> str:
        return "plugin_passthrough"

    def chunk(self, documents):
        return list(documents)


def test_register_chunking_strategy_plugin() -> None:
    register_chunking_strategy_plugin("plugin_passthrough", _PassthroughPluginStrategy)
    assert "plugin_passthrough" in list_chunking_strategy_plugins()
    registry = ChunkingStrategyRegistry()
    count = apply_chunking_strategy_plugins(registry)
    assert count >= 1
    engine = ChunkingEngine(registry)
    out = engine.chunk([Document(page_content="hello")], "plugin_passthrough")
    assert len(out) == 1


def test_assert_rag_profile_wiring_requires_llm_for_contextual_enrich() -> None:
    profile = RagProfile(contextual_enrich="on")
    with pytest.raises(ValueError, match="contextual_enrich_requires_llm"):
        assert_rag_profile_wiring(profile, llm_available=False)


def test_assert_rag_profile_wiring_allows_default_harness_profile() -> None:
    assert_rag_profile_wiring(RagProfile(), llm_available=False)
