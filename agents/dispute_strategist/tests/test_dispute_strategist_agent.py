# © Artur Czarnecki. All rights reserved.

import pytest

from dispute_strategist.dispute_strategist_agent import DisputeStrategistAgent
from dispute_strategist.contract import build_agent_contract
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus
from testing_support.builder import canonical_execution_identity_scope


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_dispute_strategist_agent_typed_run_smoke():
    agent = DisputeStrategistAgent()
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
