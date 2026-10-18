# © Artur Czarnecki. All rights reserved.

"""Architecture gates — reflex strict external-work admission (PROD-Q-R1-R7-Q IMPL-01)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from external_contractor_adapter.external_contractor_adapter_agent import (
    ExternalContractorAdapterAgent,
)
from external_contractor_adapter.steps.domain_job import run_domain_job
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from intergrax.contracts.agent_step_context import AgentStepContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_FACTORY = _REPO / "applications" / "governed_contractor_application" / "host" / "factory.py"
_DOMAIN_JOB = (
    _REPO / "agents" / "external_contractor_adapter" / "steps" / "domain_job.py"
)


def test_strict_gca_factory_wires_sanctioned_external_work_composition() -> None:
    text = _FACTORY.read_text(encoding="utf-8")
    assert "build_strict_external_work_host_session_resources" in text
    assert "attach_sanctioned_external_work_to_host_runtime" in text
    assert "build_strict_external_work_host_session_resources" in text
    assert "require_sanctioned_external_work_mutations" not in text or (
        "attach_sanctioned_external_work_to_host_runtime" in text
    )


def test_domain_job_strict_path_does_not_import_applications_package() -> None:
    tree = ast.parse(_DOMAIN_JOB.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("applications.")
            assert not node.module.startswith("governed_contractor_application.")


@pytest.mark.asyncio
async def test_strict_domain_job_fail_closed_without_sanctioned_port() -> None:
    fake = DeterministicExternalWorkFake()
    step = AgentStepContext(
        tenant_id="t",
        task_id="task-1",
        run_id="run-1",
        message="mutate",
        metadata={"external_work.tenant_id": "t", "external_work.principal_id": "p"},
    )
    out = await run_domain_job(
        step,
        external_work=fake,
        require_sanctioned_external_work_mutations=True,
        sanctioned_operation_port=None,
    )
    summary = out.get("domain_summary")
    assert isinstance(summary, dict)
    assert summary.get("used") is False
    assert fake.create_calls == 0


@pytest.mark.asyncio
async def test_strict_agent_wiring_exposes_sanctioned_port_requirement() -> None:
    agent = ExternalContractorAdapterAgent(
        external_work=DeterministicExternalWorkFake(),
        require_sanctioned_external_work_mutations=True,
        sanctioned_operation_port=None,
    )
    step = AgentStepContext(
        tenant_id="t",
        task_id="task-2",
        run_id="run-2",
        message="mutate",
    )
    result = await agent.act(step, await agent.reason(step, await agent.perceive(step)))
    summary = result.get("domain_summary")
    assert isinstance(summary, dict)
    assert summary.get("reason") == "sanctioned_external_work_operation_port_missing"
