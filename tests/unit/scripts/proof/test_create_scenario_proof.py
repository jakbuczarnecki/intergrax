# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.proof.create_scenario_proof import (
    CANONICAL_SCENARIOS_ROOT,
    DESIGN_STAGE_FORBIDDEN_ARTIFACT_NAMES,
    DESIGN_STAGE_REQUIRED_SECTIONS,
    ScenarioDesignRequest,
    ScenarioPackageExistsError,
    ScenarioSlugError,
    build_design_readme,
    create_scenario_design_package,
    scenario_package_root,
    validate_scenario_slug,
)

pytestmark = pytest.mark.unit


def test_validate_slug_accepts_canonical_example() -> None:
    slug = validate_scenario_slug("ai_incident_investigation")
    assert slug.value == "ai_incident_investigation"


def test_validate_slug_rejects_uppercase() -> None:
    with pytest.raises(ScenarioSlugError, match="lowercase"):
        validate_scenario_slug("AI_Incident")


def test_validate_slug_rejects_path_traversal() -> None:
    with pytest.raises(ScenarioSlugError, match="path separators"):
        validate_scenario_slug("../escape")


def test_validate_slug_rejects_hyphenated_slug() -> None:
    with pytest.raises(ScenarioSlugError, match="lowercase"):
        validate_scenario_slug("ai-incident")


def test_scenario_package_root_stays_under_scenarios_root(tmp_path: Path) -> None:
    slug = validate_scenario_slug("sample_scenario")
    package_root = scenario_package_root(tmp_path, slug)
    scenarios_root = (tmp_path / CANONICAL_SCENARIOS_ROOT).resolve()
    assert package_root == scenarios_root / "sample_scenario"
    assert package_root.resolve().relative_to(scenarios_root)


def test_create_scenario_design_package_generates_canonical_path(tmp_path: Path) -> None:
    slug = validate_scenario_slug("warehouse_sla_probe")
    title = "Warehouse SLA Probe"
    package = create_scenario_design_package(
        ScenarioDesignRequest(slug=slug, title=title, repo_root=tmp_path),
    )
    expected = tmp_path / CANONICAL_SCENARIOS_ROOT / "warehouse_sla_probe"
    assert package.package_root == expected
    assert package.readme_path == expected / "README.md"
    assert package.readme_path.is_file()


def test_generated_readme_contains_required_sections(tmp_path: Path) -> None:
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug("section_contract"),
            title="Section Contract",
            repo_root=tmp_path,
        ),
    )
    readme = package.readme_path.read_text(encoding="utf-8")
    for section in DESIGN_STAGE_REQUIRED_SECTIONS:
        assert section in readme


def test_existing_target_is_never_overwritten(tmp_path: Path) -> None:
    slug = validate_scenario_slug("duplicate_guard")
    request = ScenarioDesignRequest(
        slug=slug,
        title="First",
        repo_root=tmp_path,
    )
    create_scenario_design_package(request)
    with pytest.raises(ScenarioPackageExistsError, match="already exists"):
        create_scenario_design_package(request)


def test_design_package_creates_no_runtime_artifacts(tmp_path: Path) -> None:
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug("design_only"),
            title="Design Only",
            repo_root=tmp_path,
        ),
    )
    created_names = {path.name for path in package.package_root.iterdir()}
    assert created_names == {"README.md"}
    for forbidden in DESIGN_STAGE_FORBIDDEN_ARTIFACT_NAMES:
        assert not (package.package_root / forbidden).exists()
    assert not (package.package_root / "fixtures").exists()
    assert not (package.package_root / "output").exists()


def test_build_design_readme_is_deterministic() -> None:
    first = build_design_readme("Same Title")
    second = build_design_readme("Same Title")
    assert first == second


def test_scenario_one_slug_generated_by_scaffold(tmp_path: Path) -> None:
    slug = validate_scenario_slug("ai_incident_investigation")
    title = "AI Incident Investigation with Independent Verification"
    package = create_scenario_design_package(
        ScenarioDesignRequest(slug=slug, title=title, repo_root=tmp_path),
    )
    readme = package.readme_path.read_text(encoding="utf-8")
    assert title in readme
    assert package.package_root.parent.name == "scenarios"
