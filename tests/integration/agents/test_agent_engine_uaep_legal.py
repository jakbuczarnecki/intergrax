# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.agents import supports_uaep
from intergrax.runtime.registry.agent_registry import AgentRegistry
from legal.legal_agent import LegalAgent


@pytest.mark.gate
def test_legal_agent_supports_uaep() -> None:
    assert supports_uaep(LegalAgent()) is True


@pytest.mark.gate
def test_legal_agent_registers_in_registry() -> None:
    registry = AgentRegistry()
    registry.register(LegalAgent())
    agent = registry.get("legal")
    assert agent is not None
    assert agent.get_contract().capabilities == ["legal.review"]
