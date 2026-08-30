# © Artur Czarnecki. All rights reserved.

"""Destructive scaffold conformance proof for Scenario Platform (PLATFORM-6C)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.proof.create_scenario_proof import (
    ScenarioDesignRequest,
    create_scenario_design_package,
    validate_scenario_slug,
)
from scripts.proof.init_scenario_implementation import (
    ScenarioImplementationRequest,
    init_scenario_implementation,
)
from scripts.proof.scenario_architecture_conformance import (
    ScenarioArchitectureConformanceError,
    ScenarioArchitectureRuleId,
    assert_all_initialized_scenario_architectures,
    assert_scenario_application_architecture,
    discover_initialized_scenario_slugs,
    validate_scenario_application_architecture,
)
from scripts.proof.scenario_lifecycle import (
    ScenarioGapDecisionStatus,
    ScenarioGateStatus,
    ScenarioImplementationStatus,
    ScenarioLifecycle,
    ScenarioLifecycleMetadata,
    load_scenario_lifecycle_metadata,
    write_scenario_spec_frontmatter,
)

pytestmark = pytest.mark.unit

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


def _init_clean_scaffold(tmp_path: Path, slug: str) -> Path:
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug(slug),
            title=slug.replace("_", " ").title(),
            repo_root=tmp_path,
        ),
    )
    write_scenario_spec_frontmatter(package.scenario_spec_path, _accepted_metadata(slug))
    init_scenario_implementation(
        ScenarioImplementationRequest(
            slug=validate_scenario_slug(slug),
            repo_root=tmp_path,
        ),
    )
    return package.package_root


def _append_application_source(package_root: Path, relative_path: str, suffix: str) -> None:
    path = package_root / relative_path
    path.write_text(path.read_text(encoding="utf-8") + suffix, encoding="utf-8")


def _replace_application_source(package_root: Path, relative_path: str, source: str) -> None:
    path = package_root / relative_path
    path.write_text(source, encoding="utf-8")


def _assert_rule_rejected(
    tmp_path: Path,
    package_root: Path,
    *,
    expected_rule: ScenarioArchitectureRuleId,
) -> ScenarioArchitectureRuleId:
    report = validate_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug=package_root.name,
        package_root=package_root,
    )
    assert not report.ok
    observed = next(
        (violation.rule_id for violation in report.violations if violation.rule_id is expected_rule),
        None,
    )
    assert observed is expected_rule
    return observed


@dataclass(frozen=True, slots=True)
class _DestructiveMutation:
    slug_suffix: str
    expected_rule: ScenarioArchitectureRuleId
    apply: Callable[[Path, str], None]


_DESTRUCTIVE_MUTATIONS: tuple[_DestructiveMutation, ...] = (
    _DestructiveMutation(
        slug_suffix="fixtures_import",
        expected_rule=ScenarioArchitectureRuleId.APP_IMPORT_FIXTURES,
        apply=lambda package_root, slug: _append_application_source(
            package_root,
            "application/runtime_composition.py",
            f"from platform_proofs.scenarios.{slug}.fixtures.runtime import bundle\n",
        ),
    ),
    _DestructiveMutation(
        slug_suffix="graph_executor",
        expected_rule=ScenarioArchitectureRuleId.FORBIDDEN_EXECUTION_SYMBOL,
        apply=lambda package_root, slug: _append_application_source(
            package_root,
            "application/runtime_composition.py",
            "from intergrax.runtime.nexus.engine.graph_executor import GraphExecutor\nGraphExecutor\n",
        ),
    ),
    _DestructiveMutation(
        slug_suffix="mint_execution_id",
        expected_rule=ScenarioArchitectureRuleId.FORBIDDEN_EXECUTION_SYMBOL,
        apply=lambda package_root, slug: _append_application_source(
            package_root,
            "application/scenario.py",
            "from intergrax.runtime.execution.identity import mint_execution_id\nmint_execution_id()\n",
        ),
    ),
    _DestructiveMutation(
        slug_suffix="diagnostic_orchestrator",
        expected_rule=ScenarioArchitectureRuleId.FORBIDDEN_DIAGNOSTIC_ENGINE,
        apply=lambda package_root, slug: _append_application_source(
            package_root,
            "application/observability.py",
            "from intergrax.runtime.diagnostics.orchestrator import DiagnosticOrchestrator\nDiagnosticOrchestrator\n",
        ),
    ),
    _DestructiveMutation(
        slug_suffix="private_graph_executor",
        expected_rule=ScenarioArchitectureRuleId.PRIVATE_NEXUS_ATTRIBUTE,
        apply=lambda package_root, slug: _append_application_source(
            package_root,
            "application/scenario.py",
            "def _probe(composition):\n    return composition._graph_executor\n",
        ),
    ),
    _DestructiveMutation(
        slug_suffix="conformance_bypass",
        expected_rule=ScenarioArchitectureRuleId.CONFORMANCE_BYPASS,
        apply=lambda package_root, slug: _append_application_source(
            package_root,
            "application/runtime_composition.py",
            "def _bypass():\n    return build_scenario_lab_runtime(conformance_check=False)\n",
        ),
    ),
)


def test_fresh_scaffold_positive_proof(tmp_path: Path) -> None:
    slug = "scaffold_positive_proof"
    package_root = _init_clean_scaffold(tmp_path, slug)

    metadata = load_scenario_lifecycle_metadata(package_root / "SCENARIO_SPEC.md")
    assert metadata.lifecycle is ScenarioLifecycle.IMPLEMENTATION_INITIALIZED
    assert metadata.implementation_status is ScenarioImplementationStatus.INITIALIZED
    assert (package_root / "application").is_dir()

    assert_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug=slug,
        package_root=package_root,
    )

    slugs = discover_initialized_scenario_slugs(tmp_path)
    assert slug in slugs
    assert_all_initialized_scenario_architectures(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    _DESTRUCTIVE_MUTATIONS,
    ids=[case.slug_suffix for case in _DESTRUCTIVE_MUTATIONS],
)
def test_generated_scaffold_rejects_architecture_bypass(
    tmp_path: Path,
    mutation: _DestructiveMutation,
) -> None:
    slug = f"scaffold_{mutation.slug_suffix}"
    package_root = _init_clean_scaffold(tmp_path, slug)
    mutation.apply(package_root, slug)
    observed = _assert_rule_rejected(
        tmp_path,
        package_root,
        expected_rule=mutation.expected_rule,
    )
    assert observed is mutation.expected_rule


def test_generated_scaffold_rejects_missing_runtime_baseline(tmp_path: Path) -> None:
    slug = "scaffold_baseline_missing"
    package_root = _init_clean_scaffold(tmp_path, slug)
    _replace_application_source(
        package_root,
        "application/runtime_composition.py",
        (
            '"""Custom runtime without shared scenario baseline references."""\n\n'
            "from __future__ import annotations\n\n\n"
            "def build_scenario_runtime():\n"
            "    return object()\n"
        ),
    )
    _replace_application_source(
        package_root,
        "application/scenario.py",
        (
            '"""Scenario entry without shared scenario baseline references."""\n\n'
            "from __future__ import annotations\n\n\n"
            "async def execute_scenario(*, tenant_id: str, message: str, composition=None):\n"
            "    return None\n"
        ),
    )
    observed = _assert_rule_rejected(
        tmp_path,
        package_root,
        expected_rule=ScenarioArchitectureRuleId.BASELINE_RUNTIME_MISSING,
    )
    assert observed is ScenarioArchitectureRuleId.BASELINE_RUNTIME_MISSING


def test_destructive_mutations_do_not_affect_other_cases(tmp_path: Path) -> None:
    slug = "scaffold_isolation_probe"
    package_root = _init_clean_scaffold(tmp_path, slug)
    assert_scenario_application_architecture(
        repo_root=tmp_path,
        scenario_slug=slug,
        package_root=package_root,
    )
    _append_application_source(
        package_root,
        "application/runtime_composition.py",
        "from intergrax.runtime.nexus.engine.graph_executor import GraphExecutor\n",
    )
    with pytest.raises(ScenarioArchitectureConformanceError):
        assert_scenario_application_architecture(
            repo_root=tmp_path,
            scenario_slug=slug,
            package_root=package_root,
        )
