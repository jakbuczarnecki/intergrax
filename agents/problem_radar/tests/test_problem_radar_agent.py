# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest

from problem_radar.problem_radar_agent import ProblemRadarAgent
from problem_radar.schemas.output import ProblemRadarOutput
from problem_radar.steps.domain import build_stub_problem_radar_output
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState


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
@pytest.mark.integration
@pytest.mark.gate
async def test_problem_radar_runs_through_nexus() -> None:
    registry = AgentRegistry()
    registry.register(ProblemRadarAgent(), requires_uaep=True)
    loop = NexusLoop(registry)
    result = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="devtools CI pain",
            context=TaskContext(capability="problem_radar.scan"),
        )
    )
    assert result.state == TaskState.COMPLETED
    assert result.agent_id == "problem_radar"
    payload = json.loads(result.answer)
    parsed = ProblemRadarOutput.model_validate(payload)
    assert parsed.clusters
    assert "devtools CI pain" in parsed.clusters[0].title
