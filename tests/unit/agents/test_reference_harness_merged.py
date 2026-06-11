# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.reference_harness import (
    build_lab_agent_runtime_config_from_merged,
    default_reference_harness,
)
from intergrax.agents.run_environment import EffectiveAgentRunEnvironment
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from typing import Optional, Sequence


class _StubLLM(LLMAdapter):
    provider = "stub"
    model = "stub"

    @property
    def context_window_tokens(self) -> int:
        return 4096

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        _ = messages, temperature, max_tokens, run_id
        return build_adapter_response(content="ok")


@pytest.mark.unit
@pytest.mark.gate
def test_build_runtime_config_from_merged_uses_profile_flags() -> None:
    merged = EffectiveAgentRunEnvironment(
        agent_id="echo",
        contract_id="echo",
        tenant_id="tenant-a",
        enable_rag=False,
        enable_websearch=False,
    )
    config = build_lab_agent_runtime_config_from_merged(
        request=RuntimeRequest(
            agent_id="echo",
            user_id="user-1",
            session_id="sess-1",
            message="hi",
            tenant_id="tenant-a",
        ),
        llm_adapter=_StubLLM(),
        harness=default_reference_harness(),
        merged=merged,
    )
    assert config.enable_rag is False
    assert config.enable_websearch is False
