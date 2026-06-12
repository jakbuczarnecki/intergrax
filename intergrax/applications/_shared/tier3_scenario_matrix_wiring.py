# © Artur Czarnecki. All rights reserved.

"""Tier-3 scenario matrix registry and CI validation (APP-CON-7 · §35 · §44)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Tier3ScenarioId(StrEnum):
    """Architecture §44 scenario rows."""

    REACTIVE_SINGLE_AGENT = "reactive_single_agent"
    ALWAYS_ON_DAEMON = "always_on_daemon"
    SCHEDULED_QUEUE = "scheduled_queue"
    HYBRID = "hybrid"
    MULTI_AGENT_GRAPH = "multi_agent_graph"
    VIRTUAL_ORG = "virtual_org"
    SIMULATION = "simulation"
    MUTATING_PROD = "mutating_prod"
    APPLICATION_HOST_HOOK = "application_host_hook"
    BUDGET_EXCEED = "budget_exceed"
    EVAL_CI_HARNESS = "eval_ci_harness"
    YAML_DRIVEN_LAB = "yaml_driven_lab"


class Tier3UseCaseId(StrEnum):
    """Architecture §35 UC-A* catalog."""

    UC_A1 = "UC-A1"
    UC_A2 = "UC-A2"
    UC_A3 = "UC-A3"
    UC_A4 = "UC-A4"
    UC_A5 = "UC-A5"
    UC_A6 = "UC-A6"
    UC_A7 = "UC-A7"
    UC_A8 = "UC-A8"
    UC_A9 = "UC-A9"
    UC_A10 = "UC-A10"


@dataclass(frozen=True, slots=True)
class Tier3ScenarioDefinition:
    """Scenario row with UC-A mapping and deterministic test evidence paths."""

    scenario_id: Tier3ScenarioId
    use_cases: frozenset[Tier3UseCaseId]
    evidence_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Tier3HostScenarioProfile:
    """Minimum scenario and UC-A coverage for a reference Tier-3 host."""

    package: str
    posture: str
    required_use_cases: frozenset[Tier3UseCaseId]
    required_scenarios: frozenset[Tier3ScenarioId]
    extra_evidence_paths: tuple[str, ...] = ()


SCENARIO_CATALOG: dict[Tier3ScenarioId, Tier3ScenarioDefinition] = {
    Tier3ScenarioId.REACTIVE_SINGLE_AGENT: Tier3ScenarioDefinition(
        scenario_id=Tier3ScenarioId.REACTIVE_SINGLE_AGENT,
        use_cases=frozenset({Tier3UseCaseId.UC_A1}),
        evidence_paths=("tests/unit/applications/test_manifest_conformance.py",),
    ),
    Tier3ScenarioId.ALWAYS_ON_DAEMON: Tier3ScenarioDefinition(
        scenario_id=Tier3ScenarioId.ALWAYS_ON_DAEMON,
        use_cases=frozenset({Tier3UseCaseId.UC_A3}),
        evidence_paths=("tests/unit/applications/test_fastapi_mcp.py",),
    ),
    Tier3ScenarioId.SCHEDULED_QUEUE: Tier3ScenarioDefinition(
        scenario_id=Tier3ScenarioId.SCHEDULED_QUEUE,
        use_cases=frozenset({Tier3UseCaseId.UC_A5}),
        evidence_paths=("tests/unit/applications/test_modality_celery_wiring.py",),
    ),
    Tier3ScenarioId.HYBRID: Tier3ScenarioDefinition(
        scenario_id=Tier3ScenarioId.HYBRID,
        use_cases=frozenset({Tier3UseCaseId.UC_A5}),
        evidence_paths=("tests/unit/applications/test_orchestration_environment_presets.py",),
    ),
    Tier3ScenarioId.MULTI_AGENT_GRAPH: Tier3ScenarioDefinition(
        scenario_id=Tier3ScenarioId.MULTI_AGENT_GRAPH,
        use_cases=frozenset({Tier3UseCaseId.UC_A2}),
        evidence_paths=("tests/unit/applications/test_graph_spec_to_plan.py",),
    ),
    Tier3ScenarioId.VIRTUAL_ORG: Tier3ScenarioDefinition(
        scenario_id=Tier3ScenarioId.VIRTUAL_ORG,
        use_cases=frozenset({Tier3UseCaseId.UC_A7}),
        evidence_paths=("tests/unit/applications/test_uc11_product_host_compliance.py",),
    ),
    Tier3ScenarioId.SIMULATION: Tier3ScenarioDefinition(
        scenario_id=Tier3ScenarioId.SIMULATION,
        use_cases=frozenset({Tier3UseCaseId.UC_A8}),
        evidence_paths=(
            "tests/unit/applications/test_uc11_product_host_compliance.py",
            "tests/unit/applications/test_graph_spec_to_plan.py",
        ),
    ),
    Tier3ScenarioId.MUTATING_PROD: Tier3ScenarioDefinition(
        scenario_id=Tier3ScenarioId.MUTATING_PROD,
        use_cases=frozenset({Tier3UseCaseId.UC_A6}),
        evidence_paths=(
            "tests/unit/applications/test_application_deploy_triad.py",
            "tests/unit/applications/test_operational_ownership_gate.py",
        ),
    ),
    Tier3ScenarioId.APPLICATION_HOST_HOOK: Tier3ScenarioDefinition(
        scenario_id=Tier3ScenarioId.APPLICATION_HOST_HOOK,
        use_cases=frozenset(),
        evidence_paths=("tests/unit/applications/test_application_host_wiring.py",),
    ),
    Tier3ScenarioId.BUDGET_EXCEED: Tier3ScenarioDefinition(
        scenario_id=Tier3ScenarioId.BUDGET_EXCEED,
        use_cases=frozenset({Tier3UseCaseId.UC_A6}),
        evidence_paths=("tests/unit/applications/test_budget_enforcement_conformance.py",),
    ),
    Tier3ScenarioId.EVAL_CI_HARNESS: Tier3ScenarioDefinition(
        scenario_id=Tier3ScenarioId.EVAL_CI_HARNESS,
        use_cases=frozenset({Tier3UseCaseId.UC_A9}),
        evidence_paths=("tests/unit/applications/test_lab_strict_harness.py",),
    ),
    Tier3ScenarioId.YAML_DRIVEN_LAB: Tier3ScenarioDefinition(
        scenario_id=Tier3ScenarioId.YAML_DRIVEN_LAB,
        use_cases=frozenset({Tier3UseCaseId.UC_A10}),
        evidence_paths=("tests/unit/applications/test_lab_manifest_wiring.py",),
    ),
}

REFERENCE_HOST_SCENARIO_MATRIX: dict[str, Tier3HostScenarioProfile] = {
    "poc_template_application": Tier3HostScenarioProfile(
        package="poc_template_application",
        posture="lab_single",
        required_use_cases=frozenset(
            {Tier3UseCaseId.UC_A1, Tier3UseCaseId.UC_A3},
        ),
        required_scenarios=frozenset(
            {
                Tier3ScenarioId.REACTIVE_SINGLE_AGENT,
                Tier3ScenarioId.ALWAYS_ON_DAEMON,
                Tier3ScenarioId.APPLICATION_HOST_HOOK,
            },
        ),
        extra_evidence_paths=("tests/unit/applications/test_poc_template_application.py",),
    ),
    "lab_application": Tier3HostScenarioProfile(
        package="lab_application",
        posture="lab_daemon",
        required_use_cases=frozenset(
            {
                Tier3UseCaseId.UC_A1,
                Tier3UseCaseId.UC_A2,
                Tier3UseCaseId.UC_A3,
                Tier3UseCaseId.UC_A9,
                Tier3UseCaseId.UC_A10,
            },
        ),
        required_scenarios=frozenset(
            {
                Tier3ScenarioId.REACTIVE_SINGLE_AGENT,
                Tier3ScenarioId.ALWAYS_ON_DAEMON,
                Tier3ScenarioId.MULTI_AGENT_GRAPH,
                Tier3ScenarioId.APPLICATION_HOST_HOOK,
                Tier3ScenarioId.EVAL_CI_HARNESS,
                Tier3ScenarioId.YAML_DRIVEN_LAB,
            },
        ),
    ),
    "legal_application": Tier3HostScenarioProfile(
        package="legal_application",
        posture="product_single_mutating",
        required_use_cases=frozenset(
            {Tier3UseCaseId.UC_A1, Tier3UseCaseId.UC_A6},
        ),
        required_scenarios=frozenset(
            {
                Tier3ScenarioId.REACTIVE_SINGLE_AGENT,
                Tier3ScenarioId.MUTATING_PROD,
                Tier3ScenarioId.BUDGET_EXCEED,
                Tier3ScenarioId.APPLICATION_HOST_HOOK,
            },
        ),
        extra_evidence_paths=("tests/unit/applications/test_legal_manifest_wiring.py",),
    ),
    "research_application": Tier3HostScenarioProfile(
        package="research_application",
        posture="product_multi_mutating",
        required_use_cases=frozenset(
            {Tier3UseCaseId.UC_A1, Tier3UseCaseId.UC_A2, Tier3UseCaseId.UC_A6},
        ),
        required_scenarios=frozenset(
            {
                Tier3ScenarioId.REACTIVE_SINGLE_AGENT,
                Tier3ScenarioId.MULTI_AGENT_GRAPH,
                Tier3ScenarioId.MUTATING_PROD,
                Tier3ScenarioId.BUDGET_EXCEED,
            },
        ),
        extra_evidence_paths=("tests/unit/applications/test_research_manifest_wiring.py",),
    ),
    "local_workspace_application": Tier3HostScenarioProfile(
        package="local_workspace_application",
        posture="product_multi_hybrid",
        required_use_cases=frozenset(
            {Tier3UseCaseId.UC_A2, Tier3UseCaseId.UC_A5, Tier3UseCaseId.UC_A6},
        ),
        required_scenarios=frozenset(
            {
                Tier3ScenarioId.MULTI_AGENT_GRAPH,
                Tier3ScenarioId.SCHEDULED_QUEUE,
                Tier3ScenarioId.MUTATING_PROD,
                Tier3ScenarioId.BUDGET_EXCEED,
            },
        ),
    ),
    "dispute_sim_application": Tier3HostScenarioProfile(
        package="dispute_sim_application",
        posture="product_simulation",
        required_use_cases=frozenset(
            {Tier3UseCaseId.UC_A2, Tier3UseCaseId.UC_A7, Tier3UseCaseId.UC_A8},
        ),
        required_scenarios=frozenset(
            {
                Tier3ScenarioId.MULTI_AGENT_GRAPH,
                Tier3ScenarioId.VIRTUAL_ORG,
                Tier3ScenarioId.SIMULATION,
                Tier3ScenarioId.MUTATING_PROD,
            },
        ),
    ),
    "intergrax_assistant_application": Tier3HostScenarioProfile(
        package="intergrax_assistant_application",
        posture="assistant_lab",
        required_use_cases=frozenset(
            {Tier3UseCaseId.UC_A1, Tier3UseCaseId.UC_A3},
        ),
        required_scenarios=frozenset(
            {
                Tier3ScenarioId.REACTIVE_SINGLE_AGENT,
                Tier3ScenarioId.ALWAYS_ON_DAEMON,
            },
        ),
    ),
}


def iter_reference_host_profiles() -> Iterator[Tier3HostScenarioProfile]:
    """Yield canonical reference host scenario profiles in stable order."""
    for package in sorted(REFERENCE_HOST_SCENARIO_MATRIX):
        yield REFERENCE_HOST_SCENARIO_MATRIX[package]


def _use_cases_for_scenarios(
    scenarios: frozenset[Tier3ScenarioId],
) -> frozenset[Tier3UseCaseId]:
    covered: set[Tier3UseCaseId] = set()
    for scenario_id in scenarios:
        definition = SCENARIO_CATALOG[scenario_id]
        covered.update(definition.use_cases)
    return frozenset(covered)


def _evidence_paths_for_profile(profile: Tier3HostScenarioProfile) -> set[str]:
    paths: set[str] = set(profile.extra_evidence_paths)
    for scenario_id in profile.required_scenarios:
        definition = SCENARIO_CATALOG[scenario_id]
        paths.update(definition.evidence_paths)
    return paths


def check_scenario_catalog_complete() -> list[str]:
    """Ensure every §44 scenario id has a catalog entry."""
    violations: list[str] = []
    for scenario_id in Tier3ScenarioId:
        if scenario_id not in SCENARIO_CATALOG:
            violations.append(f"scenario catalog missing definition for {scenario_id.value}")
        elif not SCENARIO_CATALOG[scenario_id].evidence_paths:
            violations.append(f"scenario {scenario_id.value}: evidence_paths must not be empty")
    return violations


def check_reference_host_scenario_matrix(repo_root: Path) -> list[str]:
    """Validate reference hosts, UC-A minimums, and scenario evidence files."""
    violations: list[str] = []
    violations.extend(check_scenario_catalog_complete())

    applications_root = repo_root / "applications"
    for profile in iter_reference_host_profiles():
        host_root = applications_root / profile.package
        if not host_root.is_dir():
            violations.append(f"{profile.package}: missing applications package directory")
            continue
        manifest_path = host_root / "manifest.py"
        if not manifest_path.is_file():
            violations.append(f"{profile.package}: missing manifest.py")

        covered_use_cases = _use_cases_for_scenarios(profile.required_scenarios)
        missing_use_cases = profile.required_use_cases - covered_use_cases
        if missing_use_cases:
            missing = ", ".join(sorted(item.value for item in missing_use_cases))
            violations.append(
                f"{profile.package}: required use cases not mapped from scenarios: {missing}",
            )

        for rel_path in sorted(_evidence_paths_for_profile(profile)):
            evidence = repo_root / rel_path
            if not evidence.is_file():
                violations.append(f"{profile.package}: missing scenario evidence {rel_path}")

    return violations


def check_tier3_scenario_matrix(repo_root: Path) -> list[str]:
    """Run all Tier-3 scenario matrix gate checks."""
    return check_reference_host_scenario_matrix(repo_root)
