# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.config import RuntimeConfig
from testing_support.agent_registry_bootstrap import (
    AgentRegistryBootstrapIdentityError,
    bootstrap_agent_registry_from_agents,
)
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _StubAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="stub",
            name="Stub",
            description="stub",
            capabilities=["stub.cap"],
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )


class _MismatchedIdAgent(Agent):
    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="canonical-id",
            name="Mismatch",
            description="mismatch",
            capabilities=["stub.cap"],
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        raise RuntimeError("not used")


@pytest.mark.unit
def test_agent_registry_register_and_lookup():
    registry = AgentRegistry()
    agent = _StubAgent()
    registry.register(agent)
    assert registry.has("stub")
    assert registry.get("stub") is agent
    assert registry.find_by_capability("stub.cap") == [agent]


@pytest.mark.unit
def test_bootstrap_agent_registry_from_agents_dict():
    agent = _StubAgent()
    registry = bootstrap_agent_registry_from_agents({"stub": agent})
    assert registry.get("stub") is agent


@pytest.mark.unit
def test_bootstrap_agent_registry_identity_mismatch_fail_closed():
    agent = _MismatchedIdAgent()
    with pytest.raises(AgentRegistryBootstrapIdentityError, match="identity mismatch"):
        bootstrap_agent_registry_from_agents({"dict-key": agent})
