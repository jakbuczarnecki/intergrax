# © Artur Czarnecki. All rights reserved.

"""ACP-MIG-7 — post-fleet-migration host binding smoke for key Tier-3 manifests."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.contracts.agent_run_enums import CognitivePattern
from dispute_sim_application.host.agent_builders import DISPUTE_SIM_AGENT_BUILDERS
from dispute_sim_application.manifest import DISPUTE_SIM_APPLICATION_MANIFEST
from lab_application.host.agent_builders import LAB_AGENT_BUILDERS
from lab_application.host.settings import LabApplicationSettings
from lab_application.host.wiring import build_lab_registry
from lab_application.manifest import build_lab_manifest
from intergrax.fastapi_core.config import ApiEnvironment
from legal_application.host.settings import LegalBackendSettings
from legal_application.host.wiring import build_legal_registry
from local_workspace_application.host.agent_builders import LOCAL_WORKSPACE_AGENT_BUILDERS
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST
from research_application.host.agent_builders import RESEARCH_AGENT_BUILDERS
from research_application.host.wiring import build_research_registry
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.gate]

MIGRATED_AGENT_IDS = frozenset(
    {
        "echo",
        "signoff_probe",
        "research",
        "research-summary",
        "legal",
        "local_indexer",
        "local_search",
        "local_synthesizer",
        "dispute_intake",
        "dispute_analyst",
        "dispute_strategist",
        "dispute_scenario",
        "organization_worker",
        "intergrax_assistant",
        "problem_radar",
        "vendor_discovery",
    }
)


def _assert_migrated_contract(agent_id: str, registry) -> None:
    contract = registry.get_contract(agent_id)
    assert contract.cognitive_pattern == CognitivePattern.REFLEX
    assert contract.pattern_version


def test_lab_manifest_registry_migrated_agents() -> None:
    settings = LabApplicationSettings(
        include_echo=True,
        include_signoff_probe=True,
        include_research=True,
        include_mock_agents=False,
    )
    registry = build_lab_registry(settings=settings)
    for agent_id in ("echo", "signoff_probe", "research", "research-summary"):
        if registry.has(agent_id):
            _assert_migrated_contract(agent_id, registry)


def test_legal_host_registry_migrated() -> None:
    settings = LegalBackendSettings(
        environment=ApiEnvironment.DEV,
        legal_product_profile="strict_legal",
        legal_llm_provider="ollama",
        legal_default_agent_id="legal-default",
        legal_route_prefix="/v1/legal",
        identity_source="body_or_context",
        cors_allow_origins=frozenset(),
        allowed_hosts=frozenset(),
        openapi_enabled_override=None,
        session_sqlite_path=None,
    )
    registry = build_legal_registry(settings)
    legal_id = "legal-default" if registry.has("legal-default") else "legal"
    assert registry.has(legal_id)
    _assert_migrated_contract(legal_id, registry)


def test_research_host_registry_migrated() -> None:
    registry = build_research_registry()
    for agent_id in ("research", "research-summary"):
        assert registry.has(agent_id)
        _assert_migrated_contract(agent_id, registry)


def test_lkw_host_registry_migrated() -> None:
    ctx = ApplicationBuildContext.for_manifest(LOCAL_WORKSPACE_APPLICATION_MANIFEST)
    registry = build_application_registry(
        LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        ctx,
        builders=LOCAL_WORKSPACE_AGENT_BUILDERS,
    )
    for agent_id in ("local_indexer", "local_search", "local_synthesizer"):
        assert registry.has(agent_id)
        _assert_migrated_contract(agent_id, registry)


def test_dsw_host_registry_migrated() -> None:
    ctx = ApplicationBuildContext.for_manifest(DISPUTE_SIM_APPLICATION_MANIFEST)
    registry = build_application_registry(
        DISPUTE_SIM_APPLICATION_MANIFEST,
        ctx,
        builders=DISPUTE_SIM_AGENT_BUILDERS,
    )
    for agent_id in (
        "dispute_intake",
        "dispute_analyst",
        "dispute_strategist",
        "dispute_scenario",
    ):
        assert registry.has(agent_id)
        _assert_migrated_contract(agent_id, registry)


def test_lab_manifest_builders_round_trip() -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    manifest = build_lab_manifest(settings)
    ctx = ApplicationBuildContext.for_manifest(manifest, settings=settings)
    registry = build_application_registry(manifest, ctx, builders=LAB_AGENT_BUILDERS)
    assert registry.has("echo")
    _assert_migrated_contract("echo", registry)


def test_research_manifest_builders_round_trip() -> None:
    ctx = ApplicationBuildContext.for_manifest(RESEARCH_APPLICATION_MANIFEST)
    registry = build_application_registry(
        RESEARCH_APPLICATION_MANIFEST,
        ctx,
        builders=RESEARCH_AGENT_BUILDERS,
    )
    for agent_id in ("research", "research-summary"):
        _assert_migrated_contract(agent_id, registry)
