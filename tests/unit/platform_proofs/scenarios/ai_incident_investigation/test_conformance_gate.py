# © Artur Czarnecki. All rights reserved.

"""PLATFORM-5B architecture gates and conformance for ai_incident scenario."""

from __future__ import annotations
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import build_runtime_bundle

import re
from pathlib import Path

import pytest

from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    _incident_lab_manifest,
    build_scenario_environment_profile,
    build_scenario_runtime_composition,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import SCENARIO_TOOL_IDS

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCENARIO_APP_DIR = _REPO_ROOT / "platform_proofs" / "scenarios" / "ai_incident_investigation" / "application"

_INCIDENT_BYPASS_PATTERNS = (
    re.compile(r"conformance_check\s*=\s*is_lab"),
    re.compile(r"allow_custom_tools\s*=\s*True"),
    re.compile(r"skip_catalog_validation\s*=\s*True"),
    re.compile(r"scenario_mode\s*=\s*True"),
)


def _application_sources() -> list[tuple[Path, str]]:
    return [
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(_SCENARIO_APP_DIR.glob("*.py"))
    ]


def test_ai_incident_application_must_not_disable_incident_specific_conformance_bypasses() -> None:
    violations: list[str] = []
    for path, source in _application_sources():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for pattern in _INCIDENT_BYPASS_PATTERNS:
            if pattern.search(source):
                violations.append(f"{rel} contains forbidden bypass pattern: {pattern.pattern}")
    assert violations == []


def test_ai_incident_manifest_declares_application_owned_tools() -> None:
    env = build_scenario_environment_profile()
    manifest = _incident_lab_manifest(env)
    declared = {declaration.tool_id for declaration in manifest.application_owned_tools}
    assert declared == set(SCENARIO_TOOL_IDS)


def test_ai_incident_lab_runtime_builds_with_default_conformance() -> None:
    from intergrax.applications._shared.scenario_runtime_baseline import (
        ScenarioLabAgentRegistration,
        build_scenario_lab_agent_registry,
    )
    from intergrax.tools.registry import ToolRegistry
    from platform_proofs.scenarios.ai_incident_investigation.application.incident_scope import (
        IncidentScope,
    )
    from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
        IncidentInvestigatorAgent,
    )
    from platform_proofs.scenarios.ai_incident_investigation.application.investigator_contract import (
        incident_investigator_contract,
    )
    from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
        ScenarioRuntimeComposition,
        build_scenario_environment_profile,
    )
    from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import (
        build_resolved_fixture,
    )
    from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
        register_scenario_tools,
    )

    operational_data = build_resolved_fixture().to_operational_data()
    registry = ToolRegistry()
    evidence_store = register_scenario_tools(registry, operational_data)
    environment = build_scenario_environment_profile()
    composition = ScenarioRuntimeComposition(
        environment=environment,
        tool_registry=registry,
    )
    investigator = IncidentInvestigatorAgent(
        registry=registry,
        station_id=operational_data.station_id,
        runtime_composition=composition,
        incident_scope=IncidentScope.from_operational_defaults(
            station_id=operational_data.station_id,
        ),
        evidence_store=evidence_store,
    )
    agent_registry = build_scenario_lab_agent_registry(
        ScenarioLabAgentRegistration(
            agent=investigator,
            contract=incident_investigator_contract(),
        )
    )
    composition = build_scenario_runtime_composition(
        registry=registry,
        composition=composition,
        agent_registry=agent_registry,
    )
    assert composition.platform.env_wiring.tool_wiring.registry.tool_ids()
    for tool_id in SCENARIO_TOOL_IDS:
        assert composition.platform.env_wiring.tool_wiring.registry.has(tool_id)


def test_build_runtime_bundle_passes_platform_conformance() -> None:
    bundle = build_runtime_bundle()
    registry = bundle.runtime_composition.platform.env_wiring.tool_wiring.registry
    for tool_id in SCENARIO_TOOL_IDS:
        assert registry.has(tool_id)
