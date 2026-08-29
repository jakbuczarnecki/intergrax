# © Artur Czarnecki. All rights reserved.

"""Repository-wide gate for initialized scenario application architecture (PLATFORM-6B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.proof.create_scenario_proof import (
    ScenarioDesignRequest,
    create_scenario_design_package,
    validate_scenario_slug,
)
from scripts.proof.scenario_architecture_conformance import (
    ScenarioArchitectureConformanceError,
    ScenarioArchitectureRuleId,
    assert_all_initialized_scenario_architectures,
    discover_initialized_scenario_slugs,
    validate_scenario_application_architecture,
)
from scripts.proof.scenario_lifecycle import (
    ScenarioGapDecisionStatus,
    ScenarioGateStatus,
    ScenarioImplementationStatus,
    ScenarioLifecycle,
    ScenarioLifecycleError,
    ScenarioLifecycleMetadata,
    write_scenario_spec_frontmatter,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[4]


def _accepted_metadata(slug: str) -> ScenarioLifecycleMetadata:
    return ScenarioLifecycleMetadata(
        scenario_slug=slug,
        lifecycle=ScenarioLifecycle.ACCEPTED_FOR_IMPLEMENTATION,
        implementation_status=ScenarioImplementationStatus.NOT_INITIALIZED,
        intergrax_fit=ScenarioGateStatus.COMPLETED,
        gap_decision=ScenarioGapDecisionStatus.RESOLVED,
        observability_contract=ScenarioGateStatus.COMPLETED,
        application_vs_proof_ownership=ScenarioGateStatus.COMPLETED,
    )


def _initialized_metadata(slug: str) -> ScenarioLifecycleMetadata:
    return ScenarioLifecycleMetadata(
        scenario_slug=slug,
        lifecycle=ScenarioLifecycle.IMPLEMENTATION_INITIALIZED,
        implementation_status=ScenarioImplementationStatus.INITIALIZED,
        intergrax_fit=ScenarioGateStatus.COMPLETED,
        gap_decision=ScenarioGapDecisionStatus.RESOLVED,
        observability_contract=ScenarioGateStatus.COMPLETED,
        application_vs_proof_ownership=ScenarioGateStatus.COMPLETED,
    )


def _write_package(
    repo_root: Path,
    *,
    slug: str,
    metadata: ScenarioLifecycleMetadata,
    application_source: str | None = None,
) -> Path:
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug(slug),
            title=slug.replace("_", " ").title(),
            repo_root=repo_root,
        ),
    )
    write_scenario_spec_frontmatter(package.scenario_spec_path, metadata)
    if application_source is not None:
        application_dir = package.package_root / "application"
        application_dir.mkdir(parents=True, exist_ok=True)
        (application_dir / "__init__.py").write_text("", encoding="utf-8")
        (application_dir / "runtime.py").write_text(application_source, encoding="utf-8")
    return package.package_root


def _valid_runtime_source() -> str:
    return (
        "from intergrax.applications._shared.scenario_runtime_profiles import "
        "build_scenario_lab_runtime\n"
        "from intergrax.applications._shared.scenario_runtime_baseline import "
        "ScenarioRuntimeComposition\n\n"
        "def build_runtime() -> ScenarioRuntimeComposition:\n"
        "    return build_scenario_lab_runtime(registry=None, tenant_id='t', scenario_slug='x')\n"
    )


def test_repo_gate_passes_for_all_discovered_initialized_scenarios() -> None:
    slugs = discover_initialized_scenario_slugs(REPO_ROOT)
    assert "ai_incident_investigation" in slugs
    assert_all_initialized_scenario_architectures(REPO_ROOT)


def test_initialized_valid_scenario_passes_in_temp_repo(tmp_path: Path) -> None:
    slug = "valid_initialized"
    _write_package(
        tmp_path,
        slug=slug,
        metadata=_initialized_metadata(slug),
        application_source=_valid_runtime_source(),
    )
    slugs = discover_initialized_scenario_slugs(tmp_path)
    assert slugs == (slug,)
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug=slug,
    )
    assert report.ok


def test_initialized_scenario_with_fixtures_import_fails(tmp_path: Path) -> None:
    slug = "invalid_initialized"
    package_root = _write_package(
        tmp_path,
        slug=slug,
        metadata=_initialized_metadata(slug),
        application_source=(
            _valid_runtime_source()
            + f"from platform_proofs.scenarios.{slug}.fixtures.runtime import x\n"
        ),
    )
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug=slug,
        package_root=package_root,
    )
    assert not report.ok
    assert any(
        violation.rule_id is ScenarioArchitectureRuleId.APP_IMPORT_FIXTURES
        for violation in report.violations
    )


def test_design_package_is_not_discovered(tmp_path: Path) -> None:
    slug = "design_only"
    _write_package(
        tmp_path,
        slug=slug,
        metadata=ScenarioLifecycleMetadata.initial_design(slug=slug),
    )
    assert discover_initialized_scenario_slugs(tmp_path) == ()


def test_accepted_without_initialization_is_not_discovered(tmp_path: Path) -> None:
    slug = "accepted_only"
    _write_package(
        tmp_path,
        slug=slug,
        metadata=_accepted_metadata(slug),
    )
    assert discover_initialized_scenario_slugs(tmp_path) == ()


def test_application_without_initialized_lifecycle_raises(tmp_path: Path) -> None:
    slug = "contradictory_package"
    _write_package(
        tmp_path,
        slug=slug,
        metadata=_accepted_metadata(slug),
        application_source=_valid_runtime_source(),
    )
    with pytest.raises(ScenarioLifecycleError, match="does not mark implementation initialized"):
        discover_initialized_scenario_slugs(tmp_path)


def test_multiple_initialized_scenarios_are_all_scanned(tmp_path: Path) -> None:
    for slug in ("first_initialized", "second_initialized"):
        _write_package(
            tmp_path,
            slug=slug,
            metadata=_initialized_metadata(slug),
            application_source=(
                _valid_runtime_source()
                + "from intergrax.runtime.nexus.engine.graph_executor import GraphExecutor\n"
            ),
        )
    with pytest.raises(ScenarioArchitectureConformanceError) as exc_info:
        assert_all_initialized_scenario_architectures(tmp_path)
    message = str(exc_info.value)
    assert "first_initialized" in message or "platform_proofs/scenarios/first_initialized" in message
    assert "second_initialized" in message or "platform_proofs/scenarios/second_initialized" in message
    assert message.count("SCENARIO_ARCH_FORBIDDEN_EXECUTION") >= 2
