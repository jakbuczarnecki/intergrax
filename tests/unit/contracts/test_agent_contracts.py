# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.validation import ValidationResult


@pytest.mark.unit
def test_agent_contract_fields():
    contract = AgentContract(
        id="test",
        name="Test",
        description="desc",
        capabilities=["test.cap"],
        risk_level=AgentRiskLevel.LOW,
    )
    assert contract.id == "test"
    assert contract.capabilities == ["test.cap"]


@pytest.mark.unit
def test_agent_execution_result():
    result = AgentExecutionResult(
        agent_id="test",
        run_id="run_1",
        status=AgentExecutionStatus.COMPLETED,
        summary="ok",
    )
    assert result.status == AgentExecutionStatus.COMPLETED


@pytest.mark.unit
def test_validation_result():
    vr = ValidationResult(valid=False, errors=["x"])
    assert not vr.valid
    assert vr.errors == ["x"]
