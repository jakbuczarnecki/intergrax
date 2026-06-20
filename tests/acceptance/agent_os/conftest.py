# © Artur Czarnecki. All rights reserved.

"""Shared fixtures for Agent OS acceptance tests."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry


@pytest.fixture(autouse=True)
def _reset_adaptive_signal_store() -> None:
    """FLOW-MAINT-03 — avoid Windows ``signals.db`` lock flake on teardown."""
    from intergrax.runtime.adaptive.signal_store import reset_default_signal_store_for_tests

    reset_default_signal_store_for_tests()
    yield
    reset_default_signal_store_for_tests()


@pytest.fixture
def echo_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


@pytest.fixture
def echo_loop(echo_registry: AgentRegistry) -> NexusLoop:
    return NexusLoop(echo_registry)
