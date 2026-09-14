# © Artur Czarnecki. All rights reserved.

"""Regression: boundary_demo skill resolution vs author contract (AS-3 / UAEP runtime)."""

from __future__ import annotations

import pytest

from attestation_demo.host.tool_wiring import wire_attestation_demo_tools
from boundary_demo.boundary_demo_agent import RECORDS_PUT_TOOL_ID, BoundaryDemoAgent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.reference_harness import LabHarnessContext
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import PolicyRulesProfile
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from testing_support.builder import build_runtime_request_for_tests, canonical_governed_execution_scope
from intergrax.skills.registry.bootstrap import register_default_skills
from intergrax.skills.registry.factory import build_registry_from_profile
from intergrax.skills.registry.profile import SkillProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _reset_global_catalog_after_boundary_demo_tests() -> None:
    yield
    from intergrax.core.catalog_bootstrap import reset_tier0_catalog_bootstrap_for_tests
    from intergrax.skills.registry.bootstrap import reset_default_skills_for_tests
    from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
    from intergrax.tools.registry.catalog import clear_tool_catalog

    clear_tool_catalog()
    reset_default_tools_bootstrap()
    reset_default_skills_for_tests()
    reset_tier0_catalog_bootstrap_for_tests()


_AGENT_ID = "boundary_demo_agent"


def _enforce_policy_harness() -> LabHarnessContext:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="boundary_demo.skill_resolution")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    return LabHarnessContext(policy_bundle=wire_policy_bundle(env))


def _build_registered_boundary_demo() -> tuple[AgentRegistry, BoundaryDemoAgent]:
    tool_wiring = wire_attestation_demo_tools(document_store=InMemoryDocumentStore())
    agent = BoundaryDemoAgent(
        harness=_enforce_policy_harness(),
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
    request = build_runtime_request_for_tests(
        seed="boundary-demo-skill",
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

    with canonical_governed_execution_scope("boundary-demo-skill"):
        result = await engine.run_with_result(request)

    assert result.status == AgentExecutionStatus.COMPLETED
    assert "tool_not_allowed" not in result.summary
    assert "records.put failed" not in result.summary
    assert "stored record attestation_demo/skill-resolution-regression" in result.summary
