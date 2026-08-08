# © Artur Czarnecki. All rights reserved.

"""PUBLIC-DOCS-COMMERCIALIZATION-6: core reader document contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "README.md"
HUB_PATH = REPO_ROOT / "docs" / "project" / "README.md"
WHY_PATH = REPO_ROOT / "docs" / "project" / "overview" / "WHY_INTERGRAX.md"
USE_CASES_PATH = REPO_ROOT / "docs" / "project" / "overview" / "USE_CASES.md"
ARCHITECTURE_OVERVIEW_PATH = REPO_ROOT / "docs" / "project" / "architecture" / "ARCHITECTURE_OVERVIEW.md"
BUILD_PATH = REPO_ROOT / "docs" / "project" / "builders" / "BUILD_WITH_INTERGRAX.md"
BUILDER_QUICKSTART_PATH = REPO_ROOT / "docs" / "project" / "builders" / "BUILDER_QUICKSTART.md"
PUBLIC_MAP_PATH = REPO_ROOT / "docs" / "project" / "community" / "PUBLIC_DOCUMENTATION_MAP.md"
PUBLIC_ARCHITECTURE_PATH = REPO_ROOT / "docs" / "project" / "maintainers" / "public-adoption" / "PUBLIC_DOCUMENTATION_ARCHITECTURE.md"
HERO_LIGHT_PATH = REPO_ROOT / "docs" / "project" / "assets" / "public" / "intergrax-hero-light.svg"
HERO_DARK_PATH = REPO_ROOT / "docs" / "project" / "assets" / "public" / "intergrax-hero-dark.svg"
CATEGORY_LIGHT_PATH = REPO_ROOT / "docs" / "project" / "assets" / "public" / "intergrax-category-map-light.svg"
CATEGORY_DARK_PATH = REPO_ROOT / "docs" / "project" / "assets" / "public" / "intergrax-category-map-dark.svg"

_LEGAL_HEADER = (
    "<!--\n"
    "© Artur Czarnecki. All rights reserved.\n"
    "Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.\n"
    "See LICENSE for permitted evaluation, collaboration, and contribution use.\n"
    "-->"
)

_READER_PATHS = (
    WHY_PATH,
    USE_CASES_PATH,
    ARCHITECTURE_OVERVIEW_PATH,
    BUILD_PATH,
    BUILDER_QUICKSTART_PATH,
)

_ARCH_OPENING = (
    "How Intergrax separates the specialized product application, its application "
    "operating layer, model or agent behavior, governed access to knowledge and tools, "
    "and reviewable evidence."
)

_BUILD_OPENING = (
    "Choose the right path to evaluate Intergrax, inspect its proof, "
    "or begin building a specialized application."
)

_FORBIDDEN_MAINTAINER_PHRASES = (
    "Freeze these concepts for public readers",
    "without copying detailed execution guides",
)

_LKW_QUICKSTART_SCRIPTS = (
    "applications\\local_workspace_application\\scripts\\run-lkw-product-quickstart-windows.bat",
    "./applications/local_workspace_application/scripts/run-lkw-product-quickstart-linux.sh",
    "./applications/local_workspace_application/scripts/run-lkw-product-quickstart-macos.sh",
)

_FORBIDDEN_INSTALL_CHAINS = (
    "git clone https://github.com/jakbuczarnecki/intergrax.git && cd intergrax",
    "uv sync --extra dev && uv run intergrax doctor",
)

_MATURITY_LIMITATION_MARKERS = (
    "real-user validation",
    "commercial validation",
    "Hybrid Ask combining indexed and authorized live evidence",
)

_MERMAID_FENCE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

_FORBIDDEN_SAVINGS_PHRASES = (
    "production-proven savings",
    "universal token reduction",
    "guaranteed token savings",
)

_FORBIDDEN_CLAIM_PHRASES = (
    "finished saas",
    "commercially validated",
    "production ready",
    "production-proven savings",
    "universal token reduction",
    "guaranteed token savings",
)

_INTERNAL_TASK_PATTERN = re.compile(
    r"(CTX-UCL-|TOKEN-10|LKW-SLACK-|GOOGLE-WORKSPACE-KNOWLEDGE-|"
    r"MSGRAPH-KNOWLEDGE-|PUBLIC-DOCS-COMMERCIALIZATION-)",
    re.IGNORECASE,
)

_PERCENT_PATTERN = re.compile(r"\d+\s*%")

_BOUNDARY_PHRASES_WHY = (
    "source-available",
    "active r&d",
    "backend product alpha",
    "real-user validation",
    "commercial validation",
)

_BOUNDARY_PHRASES_ARCH = (
    "application operating layer",
    "production-readiness",
    "responsibility boundaries",
    "evidence and provenance",
)

_BOUNDARY_PHRASES_BUILD = (
    "active r&d",
    "bounded",
    "production permission",
    "license",
)

_LINK_TARGETS_BY_DOC: dict[Path, tuple[str, ...]] = {
    WHY_PATH: (
        "../proofs/PROOFS.md",
        "../proofs/LKW_PLATFORM_PROOF.md",
        "../capabilities/token_optimization/README.md",
        "../architecture/ARCHITECTURE_OVERVIEW.md",
        "../builders/BUILD_WITH_INTERGRAX.md",
    ),
    ARCHITECTURE_OVERVIEW_PATH: (
        "../proofs/PROOFS.md",
        "../proofs/LKW_PLATFORM_PROOF.md",
        "../capabilities/token_optimization/README.md",
        "../capabilities/TOKEN_OPTIMIZATION_CLAIMS.md",
        "../technical/DOCUMENTATION_MAP.md",
        "../technical/guides/INTERGRAX_HARNESS_NARRATIVE.md",
    ),
    BUILD_PATH: (
        "../proofs/PROOFS.md",
        "../proofs/LKW_PLATFORM_PROOF.md",
        "../capabilities/token_optimization/README.md",
        "../capabilities/TOKEN_OPTIMIZATION_CLAIMS.md",
        "../community/PUBLIC_DOCUMENTATION_MAP.md",
        "EVALUATION_GUIDE.md",
        "../community/COLLABORATION.md",
        "../../../LICENSE",
        "README.md#try-lkw",
        "../technical/guides/AGENT_CREATION_GUIDE.md",
        "applications/USAGE.md",
        "../architecture/ARCHITECTURE_OVERVIEW.md",
        "../technical/DOCUMENTATION_MAP.md",
    ),
    BUILDER_QUICKSTART_PATH: (
        "BUILD_WITH_INTERGRAX.md",
        "EVALUATION_GUIDE.md",
        "../architecture/ARCHITECTURE_OVERVIEW.md",
        "../technical/applications/local_workspace_application/ARCHITECTURE.md",
        "../technical/guides/AGENT_CREATION_GUIDE.md",
        "../../../applications/USAGE.md",
        "../technical/DOCUMENTATION_MAP.md",
        "../product/lkw/QUICKSTART.md",
    ),
    USE_CASES_PATH: (
        "../proofs/PROOFS.md",
        "ROADMAP.md",
        "../builders/EVALUATION_GUIDE.md",
        "../community/PARTNERS.md",
        "../community/COLLABORATION.md",
        "../architecture/ARCHITECTURE_OVERVIEW.md",
        "../builders/BUILD_WITH_INTERGRAX.md",
        "../capabilities/token_optimization/README.md",
    ),
}

_LINK_CHECK_PATHS = (
    HUB_PATH,
    WHY_PATH,
    ARCHITECTURE_OVERVIEW_PATH,
    BUILD_PATH,
    BUILDER_QUICKSTART_PATH,
    USE_CASES_PATH,
    README_PATH,
    PUBLIC_MAP_PATH,
)


def _normalize(text: str) -> str:
    return re.sub(r"[*_`]", "", text).lower()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def why_text() -> str:
    return _read(WHY_PATH)


@pytest.fixture(scope="module")
def arch_text() -> str:
    return _read(ARCHITECTURE_OVERVIEW_PATH)


@pytest.fixture(scope="module")
def build_text() -> str:
    return _read(BUILD_PATH)


@pytest.fixture(scope="module")
def builder_quickstart_text() -> str:
    return _read(BUILDER_QUICKSTART_PATH)


@pytest.fixture(scope="module")
def readme_text() -> str:
    return _read(README_PATH)


def test_documents_exist() -> None:
    for path in _READER_PATHS:
        assert path.is_file(), f"Missing reader document: {path}"


def test_legal_header() -> None:
    for path in _READER_PATHS:
        text = _read(path)
        assert text.startswith(_LEGAL_HEADER), f"Missing legal header in {path.name}"


def test_required_h1_titles() -> None:
    assert _read(WHY_PATH).splitlines()[6].strip() == "# Why Intergrax"
    assert _read(ARCHITECTURE_OVERVIEW_PATH).splitlines()[6].strip() == "# Intergrax Architecture Overview"
    assert _read(BUILD_PATH).splitlines()[6].strip() == "# Build and Evaluate with Intergrax"
    assert _read(BUILDER_QUICKSTART_PATH).splitlines()[6].strip() == "# Build with Intergrax — Builder Quick Start"


def test_first_screen_maturity_boundaries(
    why_text: str, arch_text: str, build_text: str
) -> None:
    why_norm = _normalize(why_text)
    for phrase in _BOUNDARY_PHRASES_WHY:
        assert phrase in why_norm, f"WHY missing boundary phrase: {phrase}"

    arch_norm = _normalize(arch_text)
    for phrase in _BOUNDARY_PHRASES_ARCH:
        assert phrase in arch_norm, f"ARCHITECTURE missing boundary phrase: {phrase}"

    build_norm = _normalize(build_text)
    for phrase in _BOUNDARY_PHRASES_BUILD:
        assert phrase in build_norm, f"BUILD missing boundary phrase: {phrase}"


def test_mermaid_diagrams(why_text: str, arch_text: str, build_text: str) -> None:
    docs = {
        "WHY": why_text,
        "USE_CASES": _read(USE_CASES_PATH),
        "ARCHITECTURE": arch_text,
        "BUILD": build_text,
        "BUILDER_QUICKSTART": _read(BUILDER_QUICKSTART_PATH),
    }
    forbidden_tokens = ("classDef", "style", "%%{init", "theme", "http://", "https://")
    for name, text in docs.items():
        blocks = _MERMAID_FENCE.findall(text)
        assert len(blocks) >= 1, f"{name} must contain at least one Mermaid block"
        for block in blocks:
            for token in forbidden_tokens:
                assert token not in block, f"{name}: forbidden Mermaid token {token!r}"


def test_architecture_uses_conceptual_model_not_generic_hero(arch_text: str) -> None:
    assert "```mermaid" in arch_text
    assert "../assets/public/intergrax-hero-light.svg" not in arch_text
    assert "../assets/public/intergrax-hero-dark.svg" not in arch_text


def test_required_canonical_links() -> None:
    for path, targets in _LINK_TARGETS_BY_DOC.items():
        text = _read(path)
        for target in targets:
            assert target in text, f"{path.name} missing link target: {target}"


def test_relative_link_integrity() -> None:
    for doc_path in _LINK_CHECK_PATHS:
        base = doc_path.parent
        text = _read(doc_path)
        for _label, target in _MD_LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("#"):
                continue
            clean = target.split("#", 1)[0].strip()
            if not clean:
                continue
            resolved = (base / clean).resolve()
            assert resolved.exists(), f"Broken link in {doc_path.name}: {target}"


def test_public_terminology(why_text: str, arch_text: str, build_text: str) -> None:
    assert "Primary product proof" in why_text
    assert "Featured platform-capability proof" in why_text
    assert "PARTIAL" in why_text

    assert "Primary product proof" in arch_text
    assert "Featured platform-capability proof" in arch_text
    assert "PARTIAL" in arch_text

    assert "Primary product proof" in build_text
    assert "Featured platform-capability proof" in build_text
    assert "PARTIAL" in build_text


def test_architecture_operating_layer_contract(arch_text: str) -> None:
    normalized = " ".join(_normalize(arch_text).split())
    for phrase in (
        "specialized product application",
        "application operating layer",
        "governed execution",
        "agent and model behavior",
        "knowledge and tools",
        "policy and approval boundaries",
        "evidence and provenance",
        "business rule",
        "required identity and permissions",
        "primary next action",
        "technical documentation map",
    ):
        assert phrase in normalized, f"ARCHITECTURE missing semantic marker: {phrase}"

    assert "does not decide the product's business permissions" in normalized
    assert "selected resources only" in normalized


def test_unsupported_claims_not_positive() -> None:
    negation_markers = (
        "not ",
        "does not",
        "do not",
        "no ",
        "incomplete",
        "remain",
        "without",
        "better when",
        "not currently claim",
        "another approach",
    )
    for path in (*_READER_PATHS, HUB_PATH):
        lower = _normalize(_read(path))
        for phrase in _FORBIDDEN_CLAIM_PHRASES:
            start = 0
            while True:
                idx = lower.find(phrase, start)
                if idx == -1:
                    break
                context = lower[max(0, idx - 60) : idx + len(phrase) + 60]
                assert any(marker in context for marker in negation_markers), (
                    f"{path.name}: positive forbidden claim {phrase!r} at index {idx}"
                )
                start = idx + 1
        assert not _PERCENT_PATTERN.search(_read(path)), f"{path.name}: numeric savings percentage"


def test_no_internal_task_status_leakage() -> None:
    for path in (*_READER_PATHS, HUB_PATH):
        text = _read(path)
        match = _INTERNAL_TASK_PATTERN.search(text)
        assert match is None, f"{path.name} contains internal task ID: {match.group()}"


def test_readme_routing(readme_text: str) -> None:
    for link in (
        "WHY_INTERGRAX.md",
        "ARCHITECTURE_OVERVIEW.md",
        "BUILD_WITH_INTERGRAX.md",
        "LKW_PRODUCT_TOUR.md",
        "docs/project/community/PUBLIC_DOCUMENTATION_MAP.md",
    ):
        assert link in readme_text, f"README missing link: {link}"
    assert "Local Knowledge Workspace" in readme_text
    assert "Intergrax helps teams build" in readme_text


def test_project_documentation_hub_routing() -> None:
    assert HUB_PATH.is_file()
    text = _read(HUB_PATH)
    for route in (
        "Understand Intergrax",
        "Try LKW",
        "Review proof",
        "Build with Intergrax",
        "Review architecture",
        "product/lkw/LKW_PRODUCT_TOUR.md",
        "product/lkw/QUICKSTART.md",
        "proofs/PROOFS.md",
        "proofs/LKW_PLATFORM_PROOF.md",
        "builders/BUILDER_QUICKSTART.md",
        "architecture/ARCHITECTURE_OVERVIEW.md",
        "community/PUBLIC_DOCUMENTATION_MAP.md",
        "technical/DOCUMENTATION_MAP.md",
        "maintainers/public-adoption/README.md",
        "../../README.md",
    ):
        assert route in text, f"Documentation hub missing route: {route}"

    normalized = " ".join(_normalize(text).split())
    for phrase in ("source-available", "active r&d", "backend product alpha"):
        assert phrase in normalized
    assert "real-user validation" in normalized
    assert "commercial validation" in normalized
    assert "mixed indexed + authorized live hybrid ask remains incomplete" in normalized


def test_project_documentation_hub_is_not_a_map_duplicate() -> None:
    hub_text = _read(HUB_PATH)
    public_map_text = _read(PUBLIC_MAP_PATH)
    assert len(hub_text.splitlines()) < len(public_map_text.splitlines())
    assert "## Public documents" not in hub_text
    assert "| Document | Purpose |" not in hub_text


def test_public_map_synchronization() -> None:
    text = _read(PUBLIC_MAP_PATH)
    implemented_docs = (
        "WHY_INTERGRAX.md",
        "ARCHITECTURE_OVERVIEW.md",
        "BUILD_WITH_INTERGRAX.md",
    )

    for doc in implemented_docs:
        assert doc in text

    planned_match = re.search(
        r"^## Planned public structure\s*$.*?(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if planned_match is None:
        return

    planned_section = planned_match.group(0)
    for doc in implemented_docs:
        assert doc not in planned_section, f"{doc} still listed as planned"


def test_architecture_synchronization() -> None:
    text = _read(PUBLIC_ARCHITECTURE_PATH)
    assert "PUBLIC-DOCS-COMMERCIALIZATION-6" in text
    for doc in ("WHY_INTERGRAX.md", "ARCHITECTURE_OVERVIEW.md", "BUILD_WITH_INTERGRAX.md"):
        assert f"`{doc}`" in text
        assert "implemented" in text.lower()


def test_at_a_glance_sections(why_text: str, arch_text: str, build_text: str) -> None:
    for name, text in (
        ("WHY", why_text),
        ("ARCHITECTURE", arch_text),
        ("BUILD", build_text),
    ):
        assert "## At a glance" in text, f"{name} missing At a glance section"


def test_public_reader_opening_copy(arch_text: str, build_text: str) -> None:
    for phrase in _FORBIDDEN_MAINTAINER_PHRASES:
        assert phrase not in arch_text, f"ARCHITECTURE contains maintainer phrase: {phrase!r}"
        assert phrase not in build_text, f"BUILD contains maintainer phrase: {phrase!r}"

    assert _ARCH_OPENING in arch_text
    assert _BUILD_OPENING in build_text


def test_builder_quickstart_first_checkpoint_contract(builder_quickstart_text: str) -> None:
    normalized = " ".join(_normalize(builder_quickstart_text).split())
    for phrase in (
        "canonical first builder entry point",
        "one user workflow",
        "product/application-specific",
        "reusable cross-application foundation",
        "starting surface",
        "smallest coherent change",
        "verification",
        "setup and verification are route-owned",
        "first verify the behavior at its nearest existing contract",
        "build with intergrax",
        "evaluation guide",
        "lkw quick start",
    ):
        assert phrase in normalized, f"BUILDER_QUICKSTART missing semantic marker: {phrase}"

    assert "sibling evaluation route, not a mandatory step for every builder" in normalized
    assert "not to begin builder onboarding" in normalized
    assert "no generic project scaffold" in normalized
    assert "universal application template" in normalized
    assert "stable universal public sdk" in normalized
    assert "uv sync --extra dev" not in builder_quickstart_text
    assert "uv run intergrax doctor" not in builder_quickstart_text
    assert "uv run pytest -m gate -q" not in builder_quickstart_text


def test_readme_install_sequence(readme_text: str) -> None:
    for chain in _FORBIDDEN_INSTALL_CHAINS:
        assert chain not in readme_text, f"README chains install commands: {chain!r}"

    assert "## Try LKW" in readme_text
    for script in _LKW_QUICKSTART_SCRIPTS:
        assert script in readme_text, f"README missing LKW quickstart script: {script!r}"


def test_evidence_limitations_bulleted(readme_text: str) -> None:
    """Product-first README keeps maturity / Hybrid Ask limitations visible."""
    for limitation in _MATURITY_LIMITATION_MARKERS:
        assert limitation in readme_text, (
            f"README missing limitation marker: {limitation!r}"
        )
    lower = readme_text.lower()
    assert "incomplete" in lower or "not complete" in lower


def test_brevity() -> None:
    limits = {
        WHY_PATH: 240,
        ARCHITECTURE_OVERVIEW_PATH: 280,
        BUILD_PATH: 300,
        BUILDER_QUICKSTART_PATH: 220,
        README_PATH: 300,
    }
    for path, max_lines in limits.items():
        count = len(_read(path).splitlines())
        assert count <= max_lines, f"{path.name} has {count} lines (max {max_lines})"


def test_hero_assets_exist() -> None:
    assert HERO_LIGHT_PATH.is_file()
    assert HERO_DARK_PATH.is_file()


def test_why_problem_category_and_reader_fit(why_text: str) -> None:
    normalized = _normalize(why_text)
    for phrase in (
        "rebuild controlled knowledge access",
        "reusable governed foundation",
        "product team still owns",
        "agent framework or model api",
        "another approach may fit better",
        "usecases.md",
        "active r&d",
        "mixed indexed + authorized live hybrid ask remains incomplete",
    ):
        assert phrase in normalized, f"WHY missing reader-fit invariant: {phrase}"


def test_use_cases_workflow_fit_and_ownership_contract() -> None:
    normalized = " ".join(_normalize(_read(USE_CASES_PATH)).split())
    for phrase in (
        "does intergrax fit my workflow",
        "strongest current fit",
        "bounded technical fit",
        "not yet proven",
        "not a fit",
        "private governed knowledge workspace",
        "primary product proof",
        "backend product alpha / mvp",
        "partial",
        "product team remains responsible",
        "another approach may fit better",
        "primary next action is proofs",
        "evaluation guide",
        "evidence separation",
    ):
        assert phrase in normalized, f"USE_CASES missing reader-fit marker: {phrase}"

    assert "mixed indexed + authorized live hybrid ask" in normalized
    assert "remains incomplete" in normalized
    assert "complete live-provider access is incomplete" in normalized


def test_use_cases_does_not_track_provider_rollouts() -> None:
    normalized = " ".join(_normalize(_read(USE_CASES_PATH)).split())
    forbidden_roadmap_phrases = (
        "use cases to validate next",
        "next product fit to validate",
        "durable connected slack knowledge workflow",
        "slack as interaction surface and approved knowledge source",
        "governed google workspace knowledge inside lkw",
        "first bounded google workspace proof",
        "provider-specific rollout",
        "provider rollout expectations",
    )
    for phrase in forbidden_roadmap_phrases:
        assert phrase not in normalized, (
            f"USE_CASES contains provider roadmap detail: {phrase}"
        )


def test_why_category_map_assets_and_alt_text(why_text: str) -> None:
    assert CATEGORY_LIGHT_PATH.is_file()
    assert CATEGORY_DARK_PATH.is_file()
    assert "intergrax-category-map-light.svg" in why_text
    assert "intergrax-category-map-dark.svg" in why_text
    assert 'alt="Responsibility map comparing' in why_text


def test_why_canonical_opening(why_text: str) -> None:
    opening = _normalize("\n".join(why_text.splitlines()[:13]))
    for phrase in (
        "intergrax exists for teams building specialized agent applications",
        "controlled knowledge access",
        "reusable governed foundation",
        "active r&d",
        "bounded current evidence",
    ):
        assert phrase in opening, f"WHY opening missing semantic marker: {phrase}"
