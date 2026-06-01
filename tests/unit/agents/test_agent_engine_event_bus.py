# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.runtime.events.event_bus import RuntimeEventBus

pytestmark = pytest.mark.gate


def test_static_executor_uses_event_bus_when_provided() -> None:
    bus = RuntimeEventBus()
    executor = AgentEngine._resolve_static_executor(None, bus)
    assert executor._event_bus is bus
