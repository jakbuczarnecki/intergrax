# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from unittest.mock import MagicMock

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.pipelines.rag_step_policy import pipeline_should_include_rag_step

pytestmark = pytest.mark.gate


def test_rag_step_skipped_when_rag_disabled() -> None:
    llm = MagicMock()
    llm.provider = LLMProvider.OPENAI
    cfg = RuntimeConfig(llm_adapter=llm, enable_rag=False)
    state = MagicMock()
    state.context.config = cfg
    state.engine_plan = None
    assert pipeline_should_include_rag_step(state) is False
