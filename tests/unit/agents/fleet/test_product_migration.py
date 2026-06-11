# © Artur Czarnecki. All rights reserved.

import pytest

from dispute_intake.dispute_intake_agent import DisputeIntakeAgent
from legal.legal_agent import LegalAgent
from local_search.local_search_agent import LocalSearchAgent
from research.summary_agent import SummaryAgent
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus, CognitivePattern


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory", "fragment"),
    [
        (LegalAgent, "legal:"),
        (SummaryAgent, "summary:"),
        (LocalSearchAgent, "local_search:"),
        (DisputeIntakeAgent, "dispute_intake:"),
    ],
)
async def test_product_batch_typed_run(factory, fragment) -> None:
    agent = factory()
    assert agent.get_contract().cognitive_pattern == CognitivePattern.REFLEX
    result = await agent.run(
        AgentRunRequest(
            input="product batch smoke",
            identity=RequestIdentity(tenant_id="t1", user_id="u1"),
        )
    )
    assert result.status == AgentRunStatus.SUCCEEDED
    if isinstance(result.output, dict):
        summary = str(result.output.get("summary") or "")
    else:
        summary = str(result.output or "")
    assert fragment in summary


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_summary_prior_outputs_partition() -> None:
    agent = SummaryAgent()
    result = await agent.run(
        AgentRunRequest(
            input="header\n--- prior agent outputs ---\nfinding A\nfinding B",
            identity=RequestIdentity(tenant_id="t1", user_id="u1"),
        )
    )
    assert result.status == AgentRunStatus.SUCCEEDED
    if isinstance(result.output, dict):
        summary = str(result.output.get("summary") or "")
    else:
        summary = str(result.output or "")
    assert "finding A" in summary
