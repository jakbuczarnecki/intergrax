# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.proof.create_scenario_proof import (
    CANONICAL_SCENARIOS_ROOT,
    DESIGN_STAGE_FORBIDDEN_ARTIFACT_NAMES,
    DESIGN_STAGE_README_FORBIDDEN_SECTIONS,
    DESIGN_STAGE_README_REQUIRED_SECTIONS,
    DESIGN_STAGE_SPEC_REQUIRED_SECTIONS,
    LIFECYCLE_DESIGN_NOT_ACCEPTED,
    VISUAL_STORY_AUTHORING_HINT,
    ScenarioDesignRequest,
    ScenarioPackageExistsError,
    ScenarioSlugError,
    build_design_readme,
    build_design_scenario_spec,
    create_scenario_design_package,
    scenario_package_root,
    validate_scenario_slug,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SCENARIOS_ROOT = REPO_ROOT / CANONICAL_SCENARIOS_ROOT

# Allowed C0 controls in Scenario SVG assets: TAB, LF, CR.
_ALLOWED_SVG_CONTROL_CHARS = frozenset({0x09, 0x0A, 0x0D})


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
    assert package.scenario_spec_path == expected / "SCENARIO_SPEC.md"
    assert package.readme_path.is_file()
    assert package.scenario_spec_path.is_file()


def test_generated_readme_contains_required_gateway_sections(tmp_path: Path) -> None:
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug("section_contract"),
            title="Section Contract",
            repo_root=tmp_path,
        ),
    )
    readme = package.readme_path.read_text(encoding="utf-8")
    for section in DESIGN_STAGE_README_REQUIRED_SECTIONS:
        assert section in readme


def test_generated_scenario_spec_contains_required_deep_sections(tmp_path: Path) -> None:
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug("spec_contract"),
            title="Spec Contract",
            repo_root=tmp_path,
        ),
    )
    spec = package.scenario_spec_path.read_text(encoding="utf-8")
    for section in DESIGN_STAGE_SPEC_REQUIRED_SECTIONS:
        assert section in spec


def test_generated_readme_does_not_contain_deep_abcde_contract(tmp_path: Path) -> None:
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug("gateway_only"),
            title="Gateway Only",
            repo_root=tmp_path,
        ),
    )
    readme = package.readme_path.read_text(encoding="utf-8")
    for forbidden in DESIGN_STAGE_README_FORBIDDEN_SECTIONS:
        assert forbidden not in readme


def test_generated_readme_links_to_scenario_spec(tmp_path: Path) -> None:
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug("cross_link"),
            title="Cross Link",
            repo_root=tmp_path,
        ),
    )
    readme = package.readme_path.read_text(encoding="utf-8")
    spec = package.scenario_spec_path.read_text(encoding="utf-8")
    assert "[Read the full Scenario Specification](SCENARIO_SPEC.md)" in readme
    assert "[← Back to public Scenario page](README.md)" in spec


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
    assert created_names == {"README.md", "SCENARIO_SPEC.md"}
    for forbidden in DESIGN_STAGE_FORBIDDEN_ARTIFACT_NAMES:
        assert not (package.package_root / forbidden).exists()
    assert not (package.package_root / "fixtures").exists()
    assert not (package.package_root / "output").exists()
    assert not (package.package_root / "assets").exists()


def test_build_design_readme_is_deterministic() -> None:
    first = build_design_readme("Same Title")
    second = build_design_readme("Same Title")
    assert first == second


def test_build_design_scenario_spec_is_deterministic() -> None:
    first = build_design_scenario_spec("Same Title", slug="same_title")
    second = build_design_scenario_spec("Same Title", slug="same_title")
    assert first == second


def test_generated_scenario_spec_has_lifecycle_frontmatter(tmp_path: Path) -> None:
    slug = "frontmatter_contract"
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug(slug),
            title="Frontmatter Contract",
            repo_root=tmp_path,
        ),
    )
    spec = package.scenario_spec_path.read_text(encoding="utf-8")
    assert spec.startswith("---\n")
    assert "lifecycle: DESIGN" in spec
    assert "implementation_status: NOT_INITIALIZED" in spec
    assert f"scenario_slug: {slug}" in spec
    assert "# Scenario Specification" in spec


def test_scenario_one_slug_generated_by_scaffold(tmp_path: Path) -> None:
    slug = validate_scenario_slug("ai_incident_investigation")
    title = "AI Incident Investigation with Independent Verification"
    package = create_scenario_design_package(
        ScenarioDesignRequest(slug=slug, title=title, repo_root=tmp_path),
    )
    readme = package.readme_path.read_text(encoding="utf-8")
    assert title in readme
    assert package.package_root.parent.name == "scenarios"


def test_generated_readme_has_public_question_placeholder() -> None:
    readme = build_design_readme("Placeholder Title")
    assert "Public question" in readme
    assert "_(Public question" in readme


def test_generated_readme_has_abstract_section() -> None:
    readme = build_design_readme("Placeholder Title")
    assert "## Abstract" in readme


def test_generated_readme_has_at_a_glance_table() -> None:
    readme = build_design_readme("Placeholder Title")
    assert "## At a glance" in readme
    assert "| **Problem** |" in readme
    assert "| **Trap** |" in readme
    assert LIFECYCLE_DESIGN_NOT_ACCEPTED in readme


def test_generated_readme_has_visual_story_authoring_hint() -> None:
    readme = build_design_readme("Placeholder Title")
    assert VISUAL_STORY_AUTHORING_HINT in readme
    assert "decorative imagery" in readme


def test_generated_scenario_spec_has_conditional_authoring_prompts() -> None:
    spec = build_design_scenario_spec("Placeholder Title", slug="placeholder_title")
    assert "Hidden truth / evaluator leakage" in spec
    assert "Evidence boundary" in spec
    assert "Alternative hypotheses" in spec
    assert "Independence" in spec


def test_generated_scenario_spec_has_multi_domain_fit_prompt() -> None:
    spec = build_design_scenario_spec("Placeholder Title", slug="placeholder_title")
    assert "INTERGRAX FIT is not a single-domain assignment" in spec
    assert "participating domain(s)" in spec


def test_generated_readme_has_post_run_section_placeholders() -> None:
    readme = build_design_readme("Placeholder Title")
    assert "## Latest verified run" in readme
    assert "## Run / report / evidence / source" in readme
    assert "Not yet available" in readme


def test_scenario_one_readme_has_mandatory_abstract() -> None:
    readme_path = SCENARIOS_ROOT / "ai_incident_investigation" / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    abstract_pos = readme.index("## Abstract")
    at_a_glance_pos = readme.index("## At a glance")
    assert abstract_pos < at_a_glance_pos
    abstract_block = readme[abstract_pos:at_a_glance_pos]
    assert len(abstract_block.strip().split()) >= 40


def test_generated_scenario_spec_has_application_survival_test(tmp_path: Path) -> None:
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug("survival_test"),
            title="Survival Test",
            repo_root=tmp_path,
        ),
    )
    spec = package.scenario_spec_path.read_text(encoding="utf-8")
    assert "### Application Survival Test" in spec
    assert "Required answer: **YES**" in spec


def test_generated_scenario_spec_has_observability_contract(tmp_path: Path) -> None:
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug("obs_contract"),
            title="Obs Contract",
            repo_root=tmp_path,
        ),
    )
    spec = package.scenario_spec_path.read_text(encoding="utf-8")
    assert "### Application Observability Test" in spec
    assert "### Observability / Explainability / Diagnostics Contract" in spec
    assert "Material decisions:" in spec
    assert "Application Observability Test result:" in spec
    assert "MUST NOT be a black box" in spec
    assert "PROOF DOES NOT OWN" in spec
    assert "runtime execution trace" in spec


def test_generated_scenario_spec_has_application_vs_proof_harness(tmp_path: Path) -> None:
    package = create_scenario_design_package(
        ScenarioDesignRequest(
            slug=validate_scenario_slug("harness_split"),
            title="Harness Split",
            repo_root=tmp_path,
        ),
    )
    spec = package.scenario_spec_path.read_text(encoding="utf-8")
    assert "### APPLICATION vs PROOF HARNESS" in spec
    assert "business workflow" in spec
    assert "evaluator" in spec


def test_scenario_one_spec_contains_abcde_contract() -> None:
    spec_path = SCENARIOS_ROOT / "ai_incident_investigation" / "SCENARIO_SPEC.md"
    spec = spec_path.read_text(encoding="utf-8")
    for heading in (
        "## A. SCENARIO",
        "## B. SOLUTION",
        "## C. INTERGRAX FIT",
        "## D. GAP DECISION",
        "## E. PROOF BUILD",
    ):
        assert heading in spec


def test_scenario_one_readme_does_not_contain_abcde_headings() -> None:
    readme_path = SCENARIOS_ROOT / "ai_incident_investigation" / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    for forbidden in DESIGN_STAGE_README_FORBIDDEN_SECTIONS:
        assert forbidden not in readme


@pytest.mark.parametrize("svg_path", sorted(SCENARIOS_ROOT.rglob("*.svg")))
def test_scenario_svg_assets_have_no_forbidden_control_characters(svg_path: Path) -> None:
    raw = svg_path.read_bytes()
    text = raw.decode("utf-8")
    forbidden = [
        (index, ord(char))
        for index, char in enumerate(text)
        if ord(char) < 0x20 and ord(char) not in _ALLOWED_SVG_CONTROL_CHARS
    ]
    assert not forbidden, (
        f"{svg_path.relative_to(REPO_ROOT)} contains forbidden control characters: "
        f"{forbidden[:5]}"
    )
