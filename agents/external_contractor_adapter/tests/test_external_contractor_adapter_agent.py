# © Artur Czarnecki. All rights reserved.

import pytest

from external_contractor_adapter.external_contractor_adapter_agent import (
    ExternalContractorAdapterAgent,
)
from external_contractor_adapter.contract import build_agent_contract
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus
from testing_support.builder import canonical_execution_identity_scope


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_external_contractor_adapter_typed_run_smoke():
    agent = ExternalContractorAdapterAgent()
    contract = build_agent_contract()
    assert contract.cognitive_pattern is not None
    with canonical_execution_identity_scope("agent-smoke"):
        result = await agent.run(
            AgentRunRequest(
                input="scaffold smoke",
                identity=RequestIdentity(tenant_id="t1", user_id="u1"),
                agent_id=contract.id,
            )
        )
    assert result.status == AgentRunStatus.SUCCEEDED
    assert "external_contractor.adapt" in str(result.output)
    assert "external_work_integration_missing" in str(result.output)
