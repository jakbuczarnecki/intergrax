# © Artur Czarnecki. All rights reserved.

"""CFG-06 dispute_sim pipeline — rules routing + graph_spec sequential agents."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.orchestration_wiring import GraphSpecSeedingPlanner
from intergrax.fastapi_core.config import ApiEnvironment
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from dispute_sim_application.host.agent_builders import DISPUTE_SIM_AGENT_BUILDERS
from dispute_sim_application.host.environment_profile import build_dispute_sim_environment_profile
from dispute_sim_application.host.factory import create_dispute_sim_backend_app
from dispute_sim_application.host.settings import DisputeSimBackendSettings
from dispute_sim_application.manifest import build_dispute_sim_manifest

pytestmark = [pytest.mark.unit]

_DEV_SETTINGS = DisputeSimBackendSettings(environment=ApiEnvironment.DEV)


def test_graph_spec_seeds_only_for_pipeline_capability() -> None:
    manifest = build_dispute_sim_manifest()
    env = build_dispute_sim_environment_profile(_DEV_SETTINGS)
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=_DEV_SETTINGS,
        builders=DISPUTE_SIM_AGENT_BUILDERS,
        use_in_memory_trace=True,
    )
    planner = runtime.nexus_loop._planning_runner.planner
    assert isinstance(planner, GraphSpecSeedingPlanner)

    intake_plan = planner.plan(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="index docs",
            context=TaskContext(capability="dispute.intake"),
        ),
        runtime.registry,
    )
    assert len(intake_plan.steps) == 1
    assert intake_plan.steps[0].agent_id == "dispute_intake"

    pipeline_plan = planner.plan(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="spór",
            context=TaskContext(capability="dispute.pipeline"),
        ),
        runtime.registry,
    )
    assert len(pipeline_plan.steps) == 2
    assert pipeline_plan.steps[0].agent_id == "dispute_analyst"
    assert pipeline_plan.steps[1].agent_id == "dispute_scenario"
    assert pipeline_plan.steps[1].depends_on == [pipeline_plan.steps[0].step_id]


@pytest.mark.asyncio
async def test_dispute_sim_subcontractor_scenario_runs_two_agents() -> None:
    manifest = build_dispute_sim_manifest()
    env = build_dispute_sim_environment_profile(_DEV_SETTINGS)
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=_DEV_SETTINGS,
        builders=DISPUTE_SIM_AGENT_BUILDERS,
        use_in_memory_trace=True,
    )
    runner = UnifiedTaskRunner(runtime.nexus_loop)
    message = (
        "Mamy podwykonawcę XYZ który domaga się zapłaty za prace, ale prace są wadliwe. "
        "Wysłał pismo z wezwaniem do zapłaty. Jak mu odpisać?"
    )
    result = await runner.run_task(
        Task(
            tenant_id="org-acme",
            user_id="lawyer-1",
            message=message,
            context=TaskContext(),
            metadata={"case_id": "case-xyz-payment"},
        ),
    )
    assert result.state.value == "completed"
    assert "dispute_analyst" in result.answer
    assert "dispute_scenario" in result.answer
    agent_ids = result.metadata.get("agent_ids") or []
    assert agent_ids == ["dispute_analyst", "dispute_scenario"]


def test_dispute_sim_http_free_text_routes_to_pipeline() -> None:
    client = TestClient(create_dispute_sim_backend_app())
    response = client.post(
        "/v1/dispute_sim/run",
        json={
            "message": (
                "Podwykonawca domaga się zapłaty za wadliwe roboty. "
                "Jak przygotować odpowiedź na pismo?"
            ),
            "metadata": {"case_id": "case-xyz"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "completed"
    parsed = json.loads(body["answer"])
    assert len(parsed["agents"]) == 2
    assert parsed["agents"][0]["agent_id"] == "dispute_analyst"
    assert parsed["agents"][1]["agent_id"] == "dispute_scenario"
