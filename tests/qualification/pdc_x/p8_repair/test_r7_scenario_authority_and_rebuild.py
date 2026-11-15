# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-M / R7 (G-05, G-10): the scenario runtime resolves reference bindings and rebuilds from its own inputs."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.applications._shared.roster_agent_contract_authority import ManifestAgentContractAuthority
from intergrax.applications._shared.scenario_runtime_baseline import (
    build_scenario_runtime_from_environment,
    rebuild_scenario_runtime_from_composition,
    rewire_scenario_decision_wiring,
)
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_contract import (
    INVESTIGATOR_AGENT_ID,
    incident_investigator_contract,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import SCENARIO_TOOL_IDS
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import build_runtime_bundle
from tests.unit.platform_proofs.scenarios.ai_incident_investigation.planner_doubles import (
    ScriptedIncidentInvestigationLLM,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.fixture(autouse=True)
def _offline_scenario_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same offline resolver patches as the incident scenario conftest, scoped to this file."""
    monkeypatch.delenv("SCENARIO_AI_INCIDENT_LAB_PLANNER", raising=False)

    def _fake_resolve(*_args, agent_override=None, **_kwargs):
        return agent_override if agent_override is not None else ScriptedIncidentInvestigationLLM()

    for target in (
        "platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition.resolve_llm_adapter",
        "intergrax.applications._shared.runtime_config_bridge.resolve_llm_adapter",
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        "intergrax.applications._shared.llm_resolver.resolve_optional_llm_adapter",
        "intergrax.applications._shared.llm_resolver.resolve_optional_environment_llm_adapter",
    ):
        monkeypatch.setattr(target, _fake_resolve)


class _CountingEngine(NexusValidationEngine):
    pass


def _platform():
    return build_runtime_bundle().runtime_composition.platform


def test_the_scenario_runtime_builds_and_carries_its_rebuild_inputs() -> None:
    platform = _platform()

    assert platform.manifest is not None
    assert platform.agent_contract_authority is not None
    original = platform.application_tool_registry
    assert original is not None
    assert set(SCENARIO_TOOL_IDS) <= set(original.tool_ids())
    merged = platform.env_wiring.tool_wiring.registry
    assert set(original.tool_ids()) <= set(merged.tool_ids())


def test_without_a_contract_authority_a_reference_binding_is_still_refused(tmp_path: Path) -> None:
    platform = _platform()

    with pytest.raises(ValueError, match="AgentBinding requires agent_type or import_path"):
        build_scenario_runtime_from_environment(
            environment=platform.environment,
            registry=platform.registry,
            tenant_id=platform.tenant_id,
            manifest=platform.manifest,
            runtime_events_db_path=tmp_path / "events.db",
            trace_db_path=tmp_path / "trace.db",
            use_in_memory_trace=True,
            application_tool_registry=platform.application_tool_registry,
        )


def test_an_explicit_authority_is_forwarded_to_the_environment_wiring(tmp_path: Path) -> None:
    platform = _platform()
    authority = ManifestAgentContractAuthority(
        contracts_by_id={INVESTIGATOR_AGENT_ID: incident_investigator_contract()}
    )

    rebuilt = build_scenario_runtime_from_environment(
        environment=platform.environment,
        registry=platform.registry,
        tenant_id=platform.tenant_id,
        manifest=platform.manifest,
        runtime_events_db_path=tmp_path / "events.db",
        trace_db_path=tmp_path / "trace.db",
        use_in_memory_trace=True,
        application_tool_registry=platform.application_tool_registry,
        agent_contract_authority=authority,
    )

    assert rebuilt.agent_contract_authority is authority
    assert rebuilt.manifest is platform.manifest


def test_rewire_and_rebuild_keep_manifest_authority_and_tools() -> None:
    platform = _platform()

    rewired = rewire_scenario_decision_wiring(platform, validation_engine=_CountingEngine())
    rebuilt = rebuild_scenario_runtime_from_composition(platform, environment=platform.environment)

    for composition in (rewired, rebuilt):
        assert composition.decision_flow_gate is not None
        assert composition.manifest is platform.manifest
        assert composition.agent_contract_authority is platform.agent_contract_authority
        assert composition.application_tool_registry is platform.application_tool_registry
        for tool_id in SCENARIO_TOOL_IDS:
            assert composition.env_wiring.tool_wiring.registry.has(tool_id)
        assert not set(composition.env_wiring.tool_wiring.registry.tool_ids()) - set(
            platform.env_wiring.tool_wiring.registry.tool_ids()
        )


def test_a_composition_built_without_authority_or_registry_rebuilds_as_before(tmp_path: Path) -> None:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
    from intergrax.runtime.registry.agent_registry import AgentRegistry

    from echo.echo_agent import EchoAgent  # pyright: ignore[reportMissingImports]

    registry = AgentRegistry()
    registry.register(EchoAgent())
    manifest = ApplicationManifest.lab(
        app_id="scenario_r7_plain",
        name="R7 plain",
        route_prefix="/v1/scenario_r7_plain",
        env_prefix="SCENARIO_R7_PLAIN_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
    )
    composition = build_scenario_runtime_from_environment(
        environment=ApplicationEnvironmentProfile.lab_defaults(profile_id="scenario.r7.plain"),
        registry=registry,
        tenant_id="scenario-tenant-r7",
        manifest=manifest,
        runtime_events_db_path=tmp_path / "events.db",
        trace_db_path=tmp_path / "trace.db",
        use_in_memory_trace=True,
    )

    assert composition.manifest is not None and composition.manifest.app_id == manifest.app_id
    assert composition.agent_contract_authority is None
    assert composition.application_tool_registry is None

    rebuilt = rewire_scenario_decision_wiring(composition, validation_engine=_CountingEngine())
    assert rebuilt.decision_flow_gate is not None
    assert rebuilt.manifest is not None and rebuilt.manifest.app_id == manifest.app_id
