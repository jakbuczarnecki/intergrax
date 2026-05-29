# © Artur Czarnecki. All rights reserved.

"""Shared fixtures for Agent OS acceptance tests."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry


@pytest.fixture
def echo_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


@pytest.fixture
def echo_loop(echo_registry: AgentRegistry) -> NexusLoop:
    return NexusLoop(echo_registry)
