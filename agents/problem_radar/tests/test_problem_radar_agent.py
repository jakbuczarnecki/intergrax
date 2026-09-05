# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest

from problem_radar.problem_radar_agent import ProblemRadarAgent
from problem_radar.contract import build_agent_contract
from problem_radar.schemas.output import ProblemRadarOutput
from problem_radar.steps.domain import build_stub_problem_radar_output
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus
from testing_support.builder import canonical_execution_identity_scope


@pytest.mark.unit
def test_stub_output_is_valid_problem_radar_schema() -> None:
    report = build_stub_problem_radar_output("SaaS onboarding friction")
    assert report.clusters
    assert report.clusters[0].cluster_id == "stub-1"
    assert 0.0 <= report.confidence <= 1.0


@pytest.mark.unit
def test_contract_declares_canon_capabilities() -> None:
    contract = ProblemRadarAgent().get_contract()
    assert "problem_radar.clustering" in contract.capabilities
    assert any(s.skill_id == "research.literature_scan" for s in contract.skills)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_problem_radar_agent_typed_run_smoke() -> None:
    agent = ProblemRadarAgent()
    contract = build_agent_contract()
    with canonical_execution_identity_scope("agent-smoke"):
        result = await agent.run(
            AgentRunRequest(
                input="devtools CI pain",
                identity=RequestIdentity(tenant_id="t1", user_id="u1"),
                agent_id=contract.id,
            )
        )
    assert result.status == AgentRunStatus.SUCCEEDED
    assert "stub-1" in str(result.output)
    assert "devtools" in str(result.output).lower()
