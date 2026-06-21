# © Artur Czarnecki. All rights reserved.

"""Regression: boundary_demo skill resolution vs author contract (AS-3 / UAEP runtime)."""

from __future__ import annotations

import pytest

from attestation_demo.host.tool_wiring import wire_attestation_demo_tools
from boundary_demo.boundary_demo_agent import RECORDS_PUT_TOOL_ID, BoundaryDemoAgent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.skills.registry.bootstrap import register_default_skills
from intergrax.skills.registry.factory import build_registry_from_profile
from intergrax.skills.registry.profile import SkillProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AGENT_ID = "boundary_demo_agent"


def _build_registered_boundary_demo() -> tuple[AgentRegistry, BoundaryDemoAgent]:
    tool_wiring = wire_attestation_demo_tools(document_store=InMemoryDocumentStore())
    agent = BoundaryDemoAgent(
        tool_profile=tool_wiring.profile,
        tool_wiring_context=tool_wiring.wiring_context,
    )
    register_default_skills()
    skill_registry = build_registry_from_profile(
        SkillProfile(enabled=["data.records_admin"]),
    )
    registry = AgentRegistry()
    registry.register(
        agent,
        skill_registry=skill_registry,
        tool_registry=tool_wiring.registry,
    )
    return registry, agent


def test_boundary_demo_author_contract_does_not_predeclare_allowed_tools() -> None:
    author_contract = BoundaryDemoAgent().get_contract()
    assert author_contract.allowed_tools == []


def test_boundary_demo_registry_resolves_records_put() -> None:
    registry, _agent = _build_registered_boundary_demo()
    resolved = registry.get_contract(_AGENT_ID)
    assert RECORDS_PUT_TOOL_ID in resolved.allowed_tools


@pytest.mark.asyncio
async def test_boundary_demo_uaep_uses_registry_allowed_tools_without_author_list() -> None:
    registry, agent = _build_registered_boundary_demo()
    assert agent.get_contract().allowed_tools == []

    engine = AgentEngine(registry)
    request = RuntimeRequest(
        tenant_id="default",
        user_id="regression-user",
        session_id="regression-session",
        agent_id=_AGENT_ID,
        message="skill resolution regression",
        metadata={
            "run_id": "run_boundary_demo_skill_resolution",
            "task_id": "task_boundary_demo_skill_resolution",
            "partition_key": "attestation_demo",
            "row_key": "skill-resolution-regression",
            "record_data": {"title": "skill resolution regression", "version": 1},
        },
    )

    result = await engine.run_with_result(request)

    assert result.status == AgentExecutionStatus.COMPLETED
    assert "tool_not_allowed" not in result.summary
    assert "records.put failed" not in result.summary
    assert "stored record attestation_demo/skill-resolution-regression" in result.summary
