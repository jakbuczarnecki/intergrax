# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = pytest.mark.unit


class _NonUaepAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="non-uaep",
            name="Non UAEP",
            description="stub",
            capabilities=["stub"],
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )


def test_agent_registry_rejects_non_uaep_when_required() -> None:
    registry = AgentRegistry()
    with pytest.raises(TypeError, match="UAEPAgent"):
        registry.register(_NonUaepAgent(), requires_uaep=True)
