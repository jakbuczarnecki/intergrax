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
    ScenarioImplementationInitError,
    ScenarioImplementationRequest,
    init_scenario_implementation,
)
from scripts.proof.intergrax_platform_proof_descriptor_loader import load_descriptor
from scripts.proof.scenario_architecture_conformance import (
    assert_scenario_application_architecture,
)
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


def _assert_application_architecture_gate(package_root: Path, *, repo_root: Path) -> None:
    assert_scenario_application_architecture(
        repo_root=repo_root,
        scenario_slug=package_root.name,
        package_root=package_root,
        skip_lifecycle_check=True,
    )
    application_dir = package_root / "application"
    for path in application_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".", maxsplit=1)[0]
                    if module in _FORBIDDEN_IMPORT_MODULES:
                        pytest.fail(f"forbidden import in {path}: {alias.name}")
            if isinstance(node, ast.ImportFrom) and node.module:
                root_module = node.module.split(".", maxsplit=1)[0]
                if root_module in _FORBIDDEN_IMPORT_MODULES or node.module in _FORBIDDEN_IMPORT_MODULES:
                    pytest.fail(f"forbidden import in {path}: {node.module}")


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
    assert (package_root / "assets").is_dir()
    assert (package_root / "assets" / ".gitkeep").is_file()

    metadata = load_scenario_lifecycle_metadata(package_root / "SCENARIO_SPEC.md")
    assert metadata.lifecycle is ScenarioLifecycle.IMPLEMENTATION_INITIALIZED
    assert metadata.implementation_status is ScenarioImplementationStatus.INITIALIZED

    runtime_source = (package_root / "application" / "runtime_composition.py").read_text(
        encoding="utf-8",
    )
    assert "build_scenario_lab_runtime" in runtime_source
    assert "scenario_runtime_profiles" in runtime_source
    assert "SYNTHETIC_SCENARIO_TENANT_ID" in runtime_source
    assert "runtime_events_db_path" not in runtime_source

    scenario_source = (package_root / "application" / "scenario.py").read_text(encoding="utf-8")
    assert "ScenarioExecutionRequest" in scenario_source
    assert "execute_scenario_task" in scenario_source

    _assert_application_architecture_gate(package_root, repo_root=tmp_path)


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


def test_init_validates_architecture_before_lifecycle_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slug = "validate_before_lifecycle"
    package_root = _write_accepted_design_package(tmp_path, slug=slug)
    spec_path = package_root / "SCENARIO_SPEC.md"
    call_order: list[str] = []

    original_assert = assert_scenario_application_architecture

    def _tracking_assert(**kwargs: object) -> object:
        call_order.append("architecture")
        return original_assert(**kwargs)

    original_write = write_scenario_spec_frontmatter

    def _tracking_write(path: Path, metadata: ScenarioLifecycleMetadata) -> None:
        call_order.append("lifecycle")
        original_write(path, metadata)

    monkeypatch.setattr(
        "scripts.proof.init_scenario_implementation.assert_scenario_application_architecture",
        _tracking_assert,
    )
    monkeypatch.setattr(
        "scripts.proof.init_scenario_implementation.write_scenario_spec_frontmatter",
        _tracking_write,
    )

    init_scenario_implementation(
        ScenarioImplementationRequest(
            slug=validate_scenario_slug(slug),
            repo_root=tmp_path,
        ),
    )
    assert call_order == ["architecture", "lifecycle"]
    metadata = load_scenario_lifecycle_metadata(spec_path)
    assert metadata.lifecycle is ScenarioLifecycle.IMPLEMENTATION_INITIALIZED


def test_init_fails_when_generated_architecture_violates_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slug = "architecture_violation"
    package_root = _write_accepted_design_package(tmp_path, slug=slug)
    spec_path = package_root / "SCENARIO_SPEC.md"

    def _bad_runtime_composition(slug: str, agent_class: str) -> str:
        return (
            "from intergrax.runtime.nexus.engine.graph_executor import GraphExecutor\n"
            "GraphExecutor\n"
        )

    monkeypatch.setattr(
        "scripts.proof.init_scenario_implementation._build_runtime_composition_py",
        _bad_runtime_composition,
    )

    with pytest.raises(ScenarioImplementationInitError, match="architecture conformance"):
        init_scenario_implementation(
            ScenarioImplementationRequest(
                slug=validate_scenario_slug(slug),
                repo_root=tmp_path,
            ),
        )

    metadata = load_scenario_lifecycle_metadata(spec_path)
    assert metadata.lifecycle is ScenarioLifecycle.ACCEPTED_FOR_IMPLEMENTATION
    assert metadata.implementation_status is ScenarioImplementationStatus.NOT_INITIALIZED
    assert not (package_root / "application").exists()
    assert not (package_root / "run_proof.py").exists()
    assert not (package_root / "assets").exists()


def test_init_failure_does_not_delete_preexisting_user_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slug = "preserve_user_content"
    package_root = _write_accepted_design_package(tmp_path, slug=slug)
    user_notes = package_root / "notes.md"
    user_notes.write_text("operator notes", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.proof.init_scenario_implementation._build_runtime_composition_py",
        lambda slug, agent_class: "GraphExecutor\n",
    )

    with pytest.raises(ScenarioImplementationInitError):
        init_scenario_implementation(
            ScenarioImplementationRequest(
                slug=validate_scenario_slug(slug),
                repo_root=tmp_path,
            ),
        )

    assert user_notes.read_text(encoding="utf-8") == "operator notes"

