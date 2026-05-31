# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

import pytest
from pydantic import BaseModel

from intergrax.tools.core.contracts import (
    ToolContract,
    ToolRetryPolicy,
    ToolRiskLevel,
)

pytestmark = pytest.mark.unit


class _In(BaseModel):
    x: int


class _Out(BaseModel):
    y: int


def test_tool_contract_defaults_backward_compatible():
    contract = ToolContract(
        tool_id="demo",
        name="demo",
        description="Long description for the model.",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
    )

    assert contract.risk_level is ToolRiskLevel.LOW
    assert contract.timeout_ms == 30_000
    assert contract.retry_policy == ToolRetryPolicy()
    assert contract.injects_context is False
    assert contract.category == ""
    assert contract.tags == ()
    assert contract.description_short is None
    assert contract.llm_description() == "Long description for the model."


def test_tool_contract_llm_description_prefers_short_when_compact():
    contract = ToolContract(
        tool_id="demo",
        name="demo",
        description="Long description for the model.",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
        description_short="Short.",
    )

    assert contract.llm_description(compact=True) == "Short."
    assert contract.llm_description(compact=False) == "Long description for the model."


def test_tool_retry_policy_validates_attempts():
    with pytest.raises(ValueError, match="max_attempts"):
        ToolRetryPolicy(max_attempts=0)


def test_tool_contract_validates_timeout():
    with pytest.raises(ValueError, match="timeout_ms"):
        ToolContract(
            tool_id="demo",
            name="demo",
            description="d",
            input_schema=_In,
            output_schema=_Out,
            error_mapping={},
            side_effects=False,
            timeout_ms=0,
        )
