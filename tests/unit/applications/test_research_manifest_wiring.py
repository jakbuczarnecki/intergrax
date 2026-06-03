# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from research_application.host.agent_builders import RESEARCH_AGENT_BUILDERS
from research_application.host.wiring import build_research_registry
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_research_manifest_lists_two_agents() -> None:
    enabled = RESEARCH_APPLICATION_MANIFEST.enabled_agents()
    assert len(enabled) == 2


def test_build_research_registry_registers_pipeline_agents() -> None:
    registry = build_research_registry()
    ids = set(registry.list_agent_ids())
    assert "research" in ids
    assert "research-summary" in ids
    research_contract = registry.get_contract("research")
    assert any(s.skill_id == "research.literature_scan" for s in research_contract.skills)
    assert "rag.retrieve" in research_contract.allowed_tools
    assert "websearch.query" in research_contract.allowed_tools


def test_build_research_registry_via_builders() -> None:
    ctx = ApplicationBuildContext.for_manifest(RESEARCH_APPLICATION_MANIFEST)
    registry = build_application_registry(
        RESEARCH_APPLICATION_MANIFEST,
        ctx,
        builders=RESEARCH_AGENT_BUILDERS,
    )
    assert registry.has("research")
    assert registry.has("research-summary")
