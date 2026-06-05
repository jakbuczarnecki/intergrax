from __future__ import annotations

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskContext


class _StubAgent:
    def __init__(self, contract: AgentContract) -> None:
        self._contract = contract

    def get_contract(self) -> AgentContract:
        return self._contract

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        return CapabilityMatchResult(matched=True, score=1.0, reason="stub")


def test_agent_engine_rejects_retired_agent_in_production_mode() -> None:
    registry = AgentRegistry()
    registry.register(
        _StubAgent(
            AgentContract(
                id="retired",
                name="Retired",
                description="retired",
                capabilities=["stub.cap"],
                lifecycle_state=AgentLifecycleState.RETIRED,
            )
        )
    )
    engine = AgentEngine(registry, production_mode=True)
    request = RuntimeRequest(
        agent_id="retired",
        user_id="u1",
        session_id="s1",
        message="hello",
    )

    with pytest.raises(ValueError, match="not routable"):
        engine._resolve_agent(request)
