# © Artur Czarnecki. All rights reserved.

import pytest

from boundary_demo.boundary_demo_agent import RECORDS_PUT_TOOL_ID, BoundaryDemoAgent
from boundary_demo.capabilities import CAPABILITY
from intergrax.runtime.task.task import TaskContext


@pytest.mark.unit
@pytest.mark.gate
def test_boundary_demo_contract_smoke() -> None:
    agent = BoundaryDemoAgent()
    contract = agent.get_contract()
    assert contract.id == "boundary_demo_agent"
    assert CAPABILITY in contract.capabilities
    assert contract.allowed_tools == []
    assert contract.max_steps == 1


@pytest.mark.unit
@pytest.mark.gate
def test_boundary_demo_can_handle_capability() -> None:
    agent = BoundaryDemoAgent()
    match = agent.can_handle(TaskContext(capability=CAPABILITY))
    assert match.matched is True
    assert agent.AGENT_ID in (match.agent_id or "")


@pytest.mark.unit
@pytest.mark.gate
def test_boundary_demo_records_put_tool_id_constant() -> None:
    assert RECORDS_PUT_TOOL_ID == "records.put"
