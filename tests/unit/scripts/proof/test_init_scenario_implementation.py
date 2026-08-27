# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
import compileall
import importlib.util
import sys
from pathlib import Path

import pytest

from scripts.proof.create_scenario_proof import (
    CANONICAL_SCENARIOS_ROOT,
    ScenarioDesignRequest,
    create_scenario_design_package,
    validate_scenario_slug,
)
from scripts.proof.init_scenario_implementation import (
    ScenarioImplementationExistsError,
    ScenarioImplementationRequest,
    init_scenario_implementation,
)
from scripts.proof.intergrax_platform_proof_descriptor_loader import load_descriptor
from scripts.proof.scenario_lifecycle import (
    ScenarioGapDecisionStatus,
    ScenarioGateStatus,
    ScenarioImplementationStatus,
    ScenarioLifecycle,
    ScenarioLifecycleGateError,
    ScenarioLifecycleMetadata,
    ScenarioLifecycleParseStatus,
    load_scenario_lifecycle_metadata,
    replace_scenario_spec_frontmatter,
    validate_implementation_init_preconditions,
    write_scenario_spec_frontmatter,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]

_FORBIDDEN_IMPORT_MODULES = frozenset(
    {
        "intergrax.applications._shared.harness_host_runtime",
        "intergrax.harness.application_host",
        "scripts.proof",
    }
)
_FORBIDDEN_SYMBOLS = frozenset(
    {
        "DiagnosticOrchestrator",
        "ProblemLifecycleEngine",
        "ExecutionReconstructor",
        "GraphExecutor",
        "HarnessHostRuntime",
        "NexusLoop",
        "mint_run_id",
        "mint_attempt_id",
        "bind_active_execution_identity",
    }
)
_REQUIRED_REFERENCE = "scenario_runtime_baseline"


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


def _write_accepted_design_package(
    tmp_path: Path,
    *,
    slug: str = "accepted_scenario",
    title: str = "Accepted Scenario",
    metadata: ScenarioLifecycleMetadata | None = None,
) -> Path:
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug(slug),
            title=title,
            repo_root=tmp_path,
        ),
    )
    spec_path = package.scenario_spec_path
    accepted = metadata or _accepted_metadata(slug)
    write_scenario_spec_frontmatter(spec_path, accepted)
    return package.package_root


def _assert_application_architecture_gate(package_root: Path) -> None:
    application_dir = package_root / "application"
    for path in application_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if ".proof." in alias.name or alias.name.endswith(".proof"):
                        pytest.fail(f"proof import in application layer {path}: {alias.name}")
                    module = alias.name.split(".", maxsplit=1)[0]
                    if module in _FORBIDDEN_IMPORT_MODULES:
                        pytest.fail(f"forbidden import in {path}: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".", maxsplit=1)[0]
                    if root_module in _FORBIDDEN_IMPORT_MODULES or node.module in _FORBIDDEN_IMPORT_MODULES:
                        pytest.fail(f"forbidden import in {path}: {node.module}")
                    if ".proof." in node.module or node.module.endswith(".proof"):
                        pytest.fail(f"proof import in application layer {path}: {node.module}")
            if isinstance(node, ast.Name) and node.id in _FORBIDDEN_SYMBOLS:
                pytest.fail(f"forbidden symbol in {path}: {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_SYMBOLS:
                pytest.fail(f"forbidden symbol in {path}: {node.attr}")

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in application_dir.rglob("*.py")
    )
    assert _REQUIRED_REFERENCE in combined


def test_init_fails_when_lifecycle_is_design(tmp_path: Path) -> None:
    package_root = _write_accepted_design_package(
        tmp_path,
        slug="design_only",
        metadata=ScenarioLifecycleMetadata.initial_design(slug="design_only"),
    )
    with pytest.raises(ScenarioLifecycleGateError, match="ACCEPTED_FOR_IMPLEMENTATION"):
        init_scenario_implementation(
            ScenarioImplementationRequest(
                slug=validate_scenario_slug("design_only"),
                repo_root=tmp_path,
            ),
        )
    assert not (package_root / "application").exists()
    assert not (package_root / "run_proof.py").exists()


def test_init_fails_when_intergrax_fit_incomplete(tmp_path: Path) -> None:
    metadata = _accepted_metadata("missing_fit")
    metadata = ScenarioLifecycleMetadata(
        scenario_slug=metadata.scenario_slug,
        lifecycle=metadata.lifecycle,
        implementation_status=metadata.implementation_status,
        intergrax_fit=ScenarioGateStatus.NOT_COMPLETED,
        gap_decision=metadata.gap_decision,
        observability_contract=metadata.observability_contract,
        application_vs_proof_ownership=metadata.application_vs_proof_ownership,
    )
    package_root = _write_accepted_design_package(
        tmp_path,
        slug="missing_fit",
        metadata=metadata,
    )
    with pytest.raises(ScenarioLifecycleGateError, match="INTERGRAX FIT"):
        init_scenario_implementation(
            ScenarioImplementationRequest(
                slug=validate_scenario_slug("missing_fit"),
                repo_root=tmp_path,
            ),
        )
    assert not (package_root / "application").exists()


def test_init_fails_when_gap_decision_incomplete(tmp_path: Path) -> None:
    metadata = _accepted_metadata("missing_gap")
    metadata = ScenarioLifecycleMetadata(
        scenario_slug=metadata.scenario_slug,
        lifecycle=metadata.lifecycle,
        implementation_status=metadata.implementation_status,
        intergrax_fit=metadata.intergrax_fit,
        gap_decision=ScenarioGapDecisionStatus.NOT_COMPLETED,
        observability_contract=metadata.observability_contract,
        application_vs_proof_ownership=metadata.application_vs_proof_ownership,
    )
    package_root = _write_accepted_design_package(
        tmp_path,
        slug="missing_gap",
        metadata=metadata,
    )
    with pytest.raises(ScenarioLifecycleGateError, match="GAP DECISION"):
        init_scenario_implementation(
            ScenarioImplementationRequest(
                slug=validate_scenario_slug("missing_gap"),
                repo_root=tmp_path,
            ),
        )
    assert not (package_root / "application").exists()


def test_init_fails_when_observability_contract_incomplete(tmp_path: Path) -> None:
    metadata = _accepted_metadata("missing_obs")
    metadata = ScenarioLifecycleMetadata(
        scenario_slug=metadata.scenario_slug,
        lifecycle=metadata.lifecycle,
        implementation_status=metadata.implementation_status,
        intergrax_fit=metadata.intergrax_fit,
        gap_decision=metadata.gap_decision,
        observability_contract=ScenarioGateStatus.NOT_COMPLETED,
        application_vs_proof_ownership=metadata.application_vs_proof_ownership,
    )
    package_root = _write_accepted_design_package(
        tmp_path,
        slug="missing_obs",
        metadata=metadata,
    )
    with pytest.raises(ScenarioLifecycleGateError, match="Observability"):
        init_scenario_implementation(
            ScenarioImplementationRequest(
                slug=validate_scenario_slug("missing_obs"),
                repo_root=tmp_path,
            ),
        )
    assert not (package_root / "application").exists()


def test_init_happy_path_generates_skeleton_and_updates_lifecycle(tmp_path: Path) -> None:
    slug = "happy_path"
    package_root = _write_accepted_design_package(tmp_path, slug=slug)
    package = init_scenario_implementation(
        ScenarioImplementationRequest(
            slug=validate_scenario_slug(slug),
            repo_root=tmp_path,
        ),
    )
    assert package.package_root == package_root
    assert (package_root / "application" / "runtime_composition.py").is_file()
    assert (package_root / "proof" / "evaluator.py").is_file()
    assert (package_root / "run_proof.py").is_file()
    assert (package_root / "proof.json").is_file()

    metadata = load_scenario_lifecycle_metadata(package_root / "SCENARIO_SPEC.md")
    assert metadata.lifecycle is ScenarioLifecycle.IMPLEMENTATION_INITIALIZED
    assert metadata.implementation_status is ScenarioImplementationStatus.INITIALIZED

    runtime_source = (package_root / "application" / "runtime_composition.py").read_text(
        encoding="utf-8",
    )
    assert "build_scenario_runtime_from_environment" in runtime_source
    assert "ApplicationEnvironmentProfile.lab_defaults" in runtime_source
    assert "SYNTHETIC_SCENARIO_TENANT_ID" in runtime_source

    scenario_source = (package_root / "application" / "scenario.py").read_text(encoding="utf-8")
    assert "ScenarioExecutionRequest" in scenario_source
    assert "execute_scenario_task" in scenario_source

    _assert_application_architecture_gate(package_root)


def test_second_run_fails_without_overwrite(tmp_path: Path) -> None:
    slug = "second_run"
    _write_accepted_design_package(tmp_path, slug=slug)
    request = ScenarioImplementationRequest(
        slug=validate_scenario_slug(slug),
        repo_root=tmp_path,
    )
    first = init_scenario_implementation(request)
    first_runtime = (first.package_root / "application" / "runtime_composition.py").read_text(
        encoding="utf-8",
    )
    with pytest.raises(ScenarioImplementationExistsError, match="already exist"):
        init_scenario_implementation(request)
    second_runtime = (first.package_root / "application" / "runtime_composition.py").read_text(
        encoding="utf-8",
    )
    assert first_runtime == second_runtime


def test_generated_proof_json_passes_descriptor_loader(tmp_path: Path) -> None:
    slug = "descriptor_valid"
    package_root = _write_accepted_design_package(tmp_path, slug=slug)
    init_scenario_implementation(
        ScenarioImplementationRequest(
            slug=validate_scenario_slug(slug),
            repo_root=tmp_path,
        ),
    )
    descriptor = load_descriptor(
        package_root / "proof.json",
        repo_root=tmp_path,
    )
    assert descriptor.library_class.value == "SCENARIO"


def test_generated_python_skeleton_imports(tmp_path: Path) -> None:
    slug = "importable_skeleton"
    package_root = _write_accepted_design_package(tmp_path, slug=slug)
    init_scenario_implementation(
        ScenarioImplementationRequest(
            slug=validate_scenario_slug(slug),
            repo_root=tmp_path,
        ),
    )
    assert compileall.compile_dir(package_root, quiet=1)

    scenario_root = tmp_path / CANONICAL_SCENARIOS_ROOT
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    try:
        spec = importlib.util.spec_from_file_location(
            "generated_run_proof",
            package_root / "run_proof.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module.main)
    finally:
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))


def test_legacy_spec_without_frontmatter_fails_init(tmp_path: Path) -> None:
    slug = "legacy_spec"
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug(slug),
            title="Legacy Spec",
            repo_root=tmp_path,
        ),
    )
    legacy_body = package.scenario_spec_path.read_text(encoding="utf-8")
    if legacy_body.startswith("---"):
        legacy_body = legacy_body.split("---", maxsplit=2)[-1].lstrip("\n")
    package.scenario_spec_path.write_text(legacy_body, encoding="utf-8")

    metadata = load_scenario_lifecycle_metadata(package.scenario_spec_path, expected_slug=slug)
    assert metadata.parse_status is ScenarioLifecycleParseStatus.LEGACY
    with pytest.raises(ScenarioLifecycleGateError, match="lifecycle metadata required"):
        validate_implementation_init_preconditions(metadata)
    with pytest.raises(ScenarioLifecycleGateError, match="lifecycle metadata required"):
        init_scenario_implementation(
            ScenarioImplementationRequest(
                slug=validate_scenario_slug(slug),
                repo_root=tmp_path,
            ),
        )


def test_replace_scenario_spec_frontmatter_preserves_body() -> None:
    body = "# Scenario Specification\n\n**Scenario:** Sample\n"
    metadata = _accepted_metadata("sample")
    updated = replace_scenario_spec_frontmatter(body, metadata)
    assert updated.startswith("---\n")
    assert body.strip() in updated
