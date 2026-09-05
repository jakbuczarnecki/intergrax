# © Artur Czarnecki. All rights reserved.

import pytest

from dispute_intake.dispute_intake_agent import DisputeIntakeAgent
from dispute_intake.contract import build_agent_contract
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus
from testing_support.builder import canonical_execution_identity_scope


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_dispute_intake_agent_typed_run_smoke():
    agent = DisputeIntakeAgent()
    contract = build_agent_contract()
    with canonical_execution_identity_scope("agent-smoke"):
        result = await agent.run(
            AgentRunRequest(
                input="scaffold smoke",
                identity=RequestIdentity(tenant_id="t1", user_id="u1"),
                agent_id=contract.id,
            )
        )
    assert result.status == AgentRunStatus.SUCCEEDED
    assert "scaffold smoke" in str(result.output)
