# © Artur Czarnecki. All rights reserved.

"""Tests for universal scenario application architecture conformance (PLATFORM-6A)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.proof.scenario_architecture_conformance import (
    ScenarioArchitectureConformanceError,
    ScenarioArchitectureRuleId,
    assert_scenario_application_architecture,
    scenario_requires_application_architecture_validation,
    validate_scenario_application_architecture,
)
from scripts.proof.scenario_lifecycle import (
    ScenarioGapDecisionStatus,
    ScenarioGateStatus,
    ScenarioImplementationStatus,
    ScenarioLifecycle,
    ScenarioLifecycleMetadata,
    ScenarioLifecycleParseStatus,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def _write_application_file(
    package_root: Path,
    relative_path: str,
    source: str,
) -> Path:
    path = package_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _synthetic_package(tmp_path: Path, slug: str = "synthetic_scenario") -> Path:
    package_root = tmp_path / "platform_proofs" / "scenarios" / slug
    _write_application_file(
        package_root,
        "application/__init__.py",
        "",
    )
    return package_root


def _valid_runtime_source() -> str:
    return (
        "from intergrax.applications._shared.scenario_runtime_profiles import "
        "build_scenario_lab_runtime\n"
        "from intergrax.applications._shared.scenario_runtime_baseline import "
        "ScenarioRuntimeComposition\n\n"
        "def build_runtime() -> ScenarioRuntimeComposition:\n"
        "    return build_scenario_lab_runtime(registry=None, tenant_id='t', scenario_slug='x')\n"
    )


@pytest.mark.parametrize(
    ("source", "expected_rule"),
    [
        (
            "from platform_proofs.scenarios.synthetic_scenario.fixtures.runtime import x\n",
            ScenarioArchitectureRuleId.APP_IMPORT_FIXTURES,
        ),
        (
            "from platform_proofs.scenarios.synthetic_scenario.proof.evaluator import x\n",
            ScenarioArchitectureRuleId.APP_IMPORT_PROOF,
        ),
        (
            "from intergrax.runtime.nexus.engine.graph_executor import GraphExecutor\n",
            ScenarioArchitectureRuleId.FORBIDDEN_EXECUTION_SYMBOL,
        ),
        (
            "from intergrax.runtime.nexus.loop.nexus_loop import NexusLoop\n",
            ScenarioArchitectureRuleId.FORBIDDEN_EXECUTION_SYMBOL,
        ),
        (
            "from intergrax.runtime.execution.identity import mint_execution_id\n",
            ScenarioArchitectureRuleId.FORBIDDEN_EXECUTION_SYMBOL,
        ),
        (
            "from intergrax.runtime.execution.identity import bind_active_execution_identity\n",
            ScenarioArchitectureRuleId.FORBIDDEN_EXECUTION_SYMBOL,
        ),
        (
            "from intergrax.runtime.diagnostics.orchestrator import DiagnosticOrchestrator\n",
            ScenarioArchitectureRuleId.FORBIDDEN_DIAGNOSTIC_ENGINE,
        ),
        (
            "composition._graph_executor\n",
            ScenarioArchitectureRuleId.PRIVATE_NEXUS_ATTRIBUTE,
        ),
        (
            "composition._planner\n",
            ScenarioArchitectureRuleId.PRIVATE_NEXUS_ATTRIBUTE,
        ),
    ],
)
def test_application_violations_detected_with_rule_ids(
    tmp_path: Path,
    source: str,
    expected_rule: ScenarioArchitectureRuleId,
) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source() + "\n" + source,
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert not report.ok
    assert any(violation.rule_id is expected_rule for violation in report.violations)


def test_application_importing_own_modules_passes(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(package_root, "application/agent.py", "CONTRACT = 'agent'\n")
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source() + "from .agent import CONTRACT\n",
    )
    report = assert_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert report.ok


def test_application_importing_platform_scenario_runtime_passes(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(package_root, "application/runtime.py", _valid_runtime_source())
    report = assert_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert report.ok


def test_fixtures_importing_application_passes(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(package_root, "application/runtime.py", _valid_runtime_source())
    _write_application_file(
        package_root,
        "fixtures/runtime_bundle.py",
        "from platform_proofs.scenarios.synthetic_scenario.application.runtime import build_runtime\n",
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert report.ok


def test_proof_importing_application_passes(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(package_root, "application/runtime.py", _valid_runtime_source())
    _write_application_file(
        package_root,
        "proof/evaluator.py",
        "from platform_proofs.scenarios.synthetic_scenario.application.runtime import build_runtime\n",
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert report.ok


def test_relative_application_import_of_fixtures_fails(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source() + "from ..fixtures.runtime_bundle import build_runtime\n",
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert any(
        violation.rule_id is ScenarioArchitectureRuleId.APP_IMPORT_FIXTURES
        for violation in report.violations
    )


def test_conformance_check_false_is_forbidden(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source()
        + "def wire():\n    return build_scenario_lab_runtime(conformance_check=False)\n",
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert any(
        violation.rule_id is ScenarioArchitectureRuleId.CONFORMANCE_BYPASS
        for violation in report.violations
    )


def test_missing_baseline_runtime_reference_fails(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(package_root, "application/runtime.py", "def build_runtime():\n    return None\n")
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert any(
        violation.rule_id is ScenarioArchitectureRuleId.BASELINE_RUNTIME_MISSING
        for violation in report.violations
    )


def test_assert_raises_with_formatted_violations(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source() + "GraphExecutor\n",
    )
    with pytest.raises(ScenarioArchitectureConformanceError) as exc_info:
        assert_scenario_application_architecture(
            repo_root=tmp_path,
            scenario_slug="synthetic_scenario",
            package_root=package_root,
            skip_lifecycle_check=True,
        )
    message = str(exc_info.value)
    assert "SCENARIO_ARCH_FORBIDDEN_EXECUTION" in message
    assert "GraphExecutor" in message


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            ScenarioLifecycleMetadata(
                scenario_slug="design_only",
                lifecycle=ScenarioLifecycle.DESIGN,
                implementation_status=ScenarioImplementationStatus.NOT_INITIALIZED,
                intergrax_fit=ScenarioGateStatus.NOT_COMPLETED,
                gap_decision=ScenarioGapDecisionStatus.NOT_COMPLETED,
                observability_contract=ScenarioGateStatus.NOT_COMPLETED,
                application_vs_proof_ownership=ScenarioGateStatus.NOT_COMPLETED,
            ),
            False,
        ),
        (
            ScenarioLifecycleMetadata(
                scenario_slug="initialized",
                lifecycle=ScenarioLifecycle.IMPLEMENTATION_INITIALIZED,
                implementation_status=ScenarioImplementationStatus.INITIALIZED,
                intergrax_fit=ScenarioGateStatus.COMPLETED,
                gap_decision=ScenarioGapDecisionStatus.RESOLVED,
                observability_contract=ScenarioGateStatus.COMPLETED,
                application_vs_proof_ownership=ScenarioGateStatus.COMPLETED,
            ),
            True,
        ),
    ],
)
def test_lifecycle_applicability(
    tmp_path: Path,
    metadata: ScenarioLifecycleMetadata,
    expected: bool,
) -> None:
    package_root = _synthetic_package(tmp_path, slug=metadata.scenario_slug)
    assert (
        scenario_requires_application_architecture_validation(
            metadata,
            package_root=package_root,
        )
        is expected
    )


def test_legacy_lifecycle_metadata_skips_validation(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path, slug="legacy_scenario")
    _write_application_file(
        package_root,
        "application/runtime.py",
        "GraphExecutor\n",
    )
    metadata = ScenarioLifecycleMetadata(
        scenario_slug="legacy_scenario",
        lifecycle=ScenarioLifecycle.DESIGN,
        implementation_status=ScenarioImplementationStatus.NOT_INITIALIZED,
        intergrax_fit=ScenarioGateStatus.NOT_COMPLETED,
        gap_decision=ScenarioGapDecisionStatus.NOT_COMPLETED,
        observability_contract=ScenarioGateStatus.NOT_COMPLETED,
        application_vs_proof_ownership=ScenarioGateStatus.NOT_COMPLETED,
        parse_status=ScenarioLifecycleParseStatus.LEGACY,
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="legacy_scenario",
        package_root=package_root,
        metadata=metadata,
    )
    assert report.skipped
    assert report.ok


def test_ai_incident_investigation_passes_universal_rules() -> None:
    report = validate_scenario_application_architecture(
        repo_root=REPO_ROOT,
        scenario_slug="ai_incident_investigation",
    )
    legacy_violations = tuple(
        violation
        for violation in report.violations
        if violation.rule_id is not ScenarioArchitectureRuleId.AGENT_LIFECYCLE_BYPASS
    )
    assert not legacy_violations, "\n\n".join(
        violation.format() for violation in legacy_violations
    )


def test_agent_registry_construction_fails(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source()
        + "from intergrax.runtime.registry.agent_registry import AgentRegistry\n"
        "registry = AgentRegistry()\n",
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert any(
        violation.rule_id is ScenarioArchitectureRuleId.AGENT_LIFECYCLE_BYPASS
        for violation in report.violations
    )


def test_agent_registry_aliased_class_import_construction_fails(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source()
        + (
            "from intergrax.runtime.registry.agent_registry import AgentRegistry as AR\n"
            "registry = AR()\n"
        ),
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert any(
        violation.rule_id is ScenarioArchitectureRuleId.AGENT_LIFECYCLE_BYPASS
        for violation in report.violations
    )


def test_agent_registry_module_alias_construction_fails(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source()
        + (
            "import intergrax.runtime.registry.agent_registry as registry_module\n"
            "registry = registry_module.AgentRegistry()\n"
        ),
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert any(
        violation.rule_id is ScenarioArchitectureRuleId.AGENT_LIFECYCLE_BYPASS
        for violation in report.violations
    )


def test_agent_registry_module_alias_from_agents_fails(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source()
        + (
            "import intergrax.runtime.registry.agent_registry as registry_module\n"
            "registry = registry_module.AgentRegistry.from_agents({})\n"
        ),
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert any(
        violation.rule_id is ScenarioArchitectureRuleId.AGENT_LIFECYCLE_BYPASS
        and violation.symbol == "AgentRegistry.from_agents"
        for violation in report.violations
    )


def test_agent_registry_qualified_module_import_construction_fails(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source()
        + (
            "import intergrax.runtime.registry.agent_registry\n"
            "registry = intergrax.runtime.registry.agent_registry.AgentRegistry()\n"
        ),
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert any(
        violation.rule_id is ScenarioArchitectureRuleId.AGENT_LIFECYCLE_BYPASS
        for violation in report.violations
    )


def test_unrelated_agent_registry_attribute_access_passes(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source()
        + (
            "import some_other_package as other\n"
            "registry = other.AgentRegistry()\n"
        ),
    )
    report = assert_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert report.ok


def test_local_agent_registry_register_fails(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source()
        + (
            "from intergrax.runtime.registry.agent_registry import AgentRegistry\n"
            "from some_agent_package import SomeAgent\n\n"
            "registry = AgentRegistry()\n"
            "registry.register(SomeAgent())\n"
        ),
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    violations = [
        violation
        for violation in report.violations
        if violation.rule_id is ScenarioArchitectureRuleId.AGENT_LIFECYCLE_BYPASS
    ]
    assert len(violations) >= 2
    assert any(violation.symbol == "register" for violation in violations)


def test_agent_registry_from_agents_fails(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source()
        + (
            "from intergrax.runtime.registry.agent_registry import AgentRegistry\n"
            "registry = AgentRegistry.from_agents({})\n"
        ),
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert any(
        violation.rule_id is ScenarioArchitectureRuleId.AGENT_LIFECYCLE_BYPASS
        and violation.symbol == "AgentRegistry.from_agents"
        for violation in report.violations
    )


def test_reusable_agent_import_without_local_registry_passes(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source() + "from some_agent_package import SomeAgent\n",
    )
    report = assert_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert report.ok


def test_manifest_agent_binding_without_local_registry_passes(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source()
        + (
            "from intergrax.applications.contracts.manifest import AgentBinding\n"
            "BINDING = AgentBinding.reference(contract_id='demo.agent', capabilities=['demo.run'])\n"
        ),
    )
    report = assert_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert report.ok


def test_private_agent_registry_contracts_mutation_fails(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source()
        + (
            "from intergrax.runtime.registry.agent_registry import AgentRegistry\n"
            "registry = AgentRegistry()\n"
            "registry._contracts['demo.agent'] = object()\n"
        ),
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert any(
        violation.rule_id is ScenarioArchitectureRuleId.AGENT_LIFECYCLE_BYPASS
        and violation.symbol == "_contracts"
        for violation in report.violations
    )


def test_platform_registry_register_mutation_fails(tmp_path: Path) -> None:
    package_root = _synthetic_package(tmp_path)
    _write_application_file(
        package_root,
        "application/runtime.py",
        _valid_runtime_source()
        + (
            "def attach(agent, composition):\n"
            "    composition.platform.registry.register(agent)\n"
        ),
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug="synthetic_scenario",
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    assert any(
        violation.rule_id is ScenarioArchitectureRuleId.AGENT_LIFECYCLE_BYPASS
        and violation.symbol == "registry.register"
        for violation in report.violations
    )
