# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.profiles.runtime_rag_sync import sync_rag_profile_from_runtime_config
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

pytestmark = pytest.mark.gate


def test_sync_maps_max_docs_to_final_top_k() -> None:
    llm = MagicMock()
    llm.provider = LLMProvider.OPENAI
    cfg = RuntimeConfig(llm_adapter=llm, max_docs_per_query=12, rag_profile=RagProfile(final_top_k=4))
    profile = sync_rag_profile_from_runtime_config(cfg)
    assert profile.final_top_k == 12
