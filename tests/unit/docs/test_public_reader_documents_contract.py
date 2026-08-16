# © Artur Czarnecki. All rights reserved.

"""PUBLIC-DOCS-COMMERCIALIZATION-6: core reader document contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from intergrax.runtime.config.forbidden_generation_model_env import (
    FORBIDDEN_GENERATION_MODEL_ENV_NAMES,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "README.md"
PROOFS_PATH = REPO_ROOT / "docs" / "project" / "proofs" / "PROOFS.md"
LKW_PRODUCT_TOUR_PATH = (
    REPO_ROOT / "applications" / "local_workspace_application" / "docs" / "product" / "LKW_PRODUCT_TOUR.md"
)
LKW_QUICKSTART_PATH = (
    REPO_ROOT / "applications" / "local_workspace_application" / "docs" / "product" / "QUICKSTART.md"
)
HUB_PATH = REPO_ROOT / "docs" / "project" / "README.md"
WHY_PATH = REPO_ROOT / "docs" / "project" / "overview" / "WHY_INTERGRAX.md"
USE_CASES_PATH = REPO_ROOT / "docs" / "project" / "overview" / "USE_CASES.md"
ARCHITECTURE_OVERVIEW_PATH = REPO_ROOT / "docs" / "project" / "architecture" / "ARCHITECTURE_OVERVIEW.md"
BUILD_PATH = REPO_ROOT / "docs" / "project" / "builders" / "BUILD_WITH_INTERGRAX.md"
BUILDER_QUICKSTART_PATH = REPO_ROOT / "docs" / "project" / "builders" / "BUILDER_QUICKSTART.md"
PUBLIC_MAP_PATH = REPO_ROOT / "docs" / "project" / "community" / "PUBLIC_DOCUMENTATION_MAP.md"
PUBLIC_ARCHITECTURE_PATH = REPO_ROOT / "docs" / "project" / "maintainers" / "public-adoption" / "PUBLIC_DOCUMENTATION_ARCHITECTURE.md"
MULTIPLAYER_ARCH_PATH = (
    REPO_ROOT / "docs" / "project" / "capabilities" / "architecture" / "MULTIPLAYER_AI.md"
)
PLATFORM_PLUGINS_ARCH_PATH = REPO_ROOT / "docs" / "project" / "architecture" / "PLATFORM_PLUGINS.md"
ROADMAP_PATH = REPO_ROOT / "docs" / "project" / "overview" / "ROADMAP.md"
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
    "You have completed the [Builder Quick Start](BUILDER_QUICKSTART.md) checkpoint. "
    "Now turn the workflow into a bounded application composition plan on Intergrax."
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
    "production use",
    "license",
)

_LINK_TARGETS_BY_DOC: dict[Path, tuple[str, ...]] = {
    WHY_PATH: (
        "../proofs/PROOFS.md",
        "../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md",
        "../capabilities/token_optimization/README.md",
        "../architecture/ARCHITECTURE_OVERVIEW.md",
        "../builders/BUILD_WITH_INTERGRAX.md",
    ),
    ARCHITECTURE_OVERVIEW_PATH: (
        "../proofs/PROOFS.md",
        "../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md",
        "../capabilities/token_optimization/README.md",
        "../capabilities/TOKEN_OPTIMIZATION_CLAIMS.md",
        "../technical/DOCUMENTATION_MAP.md",
        "../technical/guides/INTERGRAX_HARNESS_NARRATIVE.md",
    ),
    BUILD_PATH: (
        "../proofs/PROOFS.md",
        "../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md",
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
        "../../../applications/local_workspace_application/docs/ARCHITECTURE.md",
        "../technical/guides/AGENT_CREATION_GUIDE.md",
        "../../../applications/USAGE.md",
        "../technical/DOCUMENTATION_MAP.md",
        "../../../applications/local_workspace_application/docs/product/QUICKSTART.md",
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
    PROOFS_PATH,
    LKW_PRODUCT_TOUR_PATH,
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
    assert _read(BUILD_PATH).splitlines()[6].strip() == "# Build With Intergrax"
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


def test_project_projections_synchronize_accepted_lkw_boundaries() -> None:
    proofs_text = _read(PROOFS_PATH)
    tour_text = _read(LKW_PRODUCT_TOUR_PATH)
    proofs = " ".join(_normalize(proofs_text).split())
    tour = " ".join(_normalize(tour_text).split())

    for text, name in ((proofs, "PROOFS"), (tour, "LKW Product Tour")):
        assert "primary product proof" in text, f"{name} omits LKW proof role"
        assert "backend product alpha / mvp" in text, f"{name} omits LKW maturity"
        assert "partial" in text, f"{name} omits LKW partial status"
        assert "real-user validation" in text, f"{name} omits real-user boundary"
        assert "commercial validation" in text, f"{name} omits commercial boundary"
        assert "complete live-provider access" in text, f"{name} omits live-provider boundary"

    assert "bounded indexed hybrid ask branch" in proofs
    assert "indexedonly" in proofs
    assert "lkw-hybrid-ask-indexed" in proofs
    assert "mixed indexed + authorized live hybrid ask remains incomplete" in proofs
    assert "actual application/runtime indexed ask path" in tour
    assert "mixed indexed + authorized live hybrid ask" in tour
    assert not re.search(r"does not represent:\s*-\s*hybrid ask\b", tour_text, re.IGNORECASE)

    assert "supporting-foundation evidence is capability-specific" in proofs
    assert not re.search(
        r"shared platform foundations.*implemented|implemented.*shared platform foundations",
        proofs,
    )
    assert "complete slack connected-knowledge product proof" not in proofs
    assert "google workspace lkw product proof" not in proofs

    assert "token optimization" in proofs
    assert "partial" in proofs
    assert "universal token reduction" in proofs
    assert "production-proven savings" in proofs
    assert not _INTERNAL_TASK_PATTERN.search(proofs_text)
    assert not _INTERNAL_TASK_PATTERN.search(tour_text)
    assert "lkw-grounded-result-light.svg" in tour_text
    assert "lkw-grounded-result-dark.svg" in tour_text


def test_lkw_quickstart_reader_literals_and_routes() -> None:
    text = _read(LKW_QUICKSTART_PATH)
    malformed_prefix = "../../"
    for literal in (
        "uv",
        "PATH",
        "INTERGRAX_ALLOWED_READ_ROOTS",
        "llama3.1:latest",
        ".env",
        ".env.example",
        "sample_docs/lkw_product_quickstart.txt",
        "127.0.0.1",
        "localhost",
        "::1",
        "status: ok",
        "--skip-stack-start",
        "docker compose",
        "applications/local_workspace_application",
    ):
        assert f"{malformed_prefix}{literal}" not in text

    assert (
        "../../"
        "applications/local_workspace_application"
    ) not in text
    for launcher in _LKW_QUICKSTART_SCRIPTS:
        assert launcher in text, f"Quick Start missing launcher: {launcher!r}"
    for marker in (
        "AURORA-17",
        "lkw_product_quickstart.txt",
        "persisted_run_verified=true",
        "[LKW Product Tour](LKW_PRODUCT_TOUR.md)",
        "[LKW Platform Proof](../proof/LKW_PLATFORM_PROOF.md)",
        "LKW-PRODUCT-QUICKSTART-WINDOWS",
        "LKW-PRODUCT-QUICKSTART-LINUX",
        "LKW-PRODUCT-QUICKSTART-MACOS",
    ):
        assert marker in text, f"Quick Start missing canonical marker: {marker!r}"

    assert "From `applications/local_workspace_application`:" in text
    assert (
        "docker compose -p intergrax_lkw -f docker/docker-compose.yml down" in text
    )

    quickstart_norm = " ".join(_normalize(text).split())
    for phrase in (
        "indexed ask v1",
        "not the separate hybrid ask proof",
        "mixed indexed + authorized-live hybrid ask",
        "not proven",
    ):
        assert phrase in quickstart_norm, (
            f"Quick Start missing Hybrid Ask boundary marker: {phrase!r}"
        )


def test_public_terminology(why_text: str, arch_text: str, build_text: str) -> None:
    assert "Primary product proof" in why_text
    assert "Featured platform-capability proof" in why_text
    assert "PARTIAL" in why_text

    assert "Primary product proof" in arch_text
    assert "Featured platform-capability proof" in arch_text
    assert "PARTIAL" in arch_text

    build_normalized = " ".join(_normalize(build_text).split())
    for phrase in (
        "concrete product workflow",
        "product responsibilities",
        "agent and model behavior",
        "knowledge and context",
        "tools and effects",
        "evidence and provenance",
        "failure and recovery",
        "nearest existing contract",
    ):
        assert phrase in build_normalized, f"BUILD missing planning marker: {phrase}"
    assert "evaluation guide" in build_normalized
    assert "proofs" in build_normalized


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
        "USE_CASES.md",
        "PROOFS.md",
        "BUILDER_QUICKSTART.md",
        "LKW_PRODUCT_TOUR.md",
        "PARTNERS.md",
        "FAQ.md",
        "docs/project/community/PUBLIC_DOCUMENTATION_MAP.md",
        "docs/project/capabilities/architecture/MULTIPLAYER_AI.md",
        "docs/project/architecture/PLATFORM_PLUGINS.md",
    ):
        assert link in readme_text, f"README missing link: {link}"
    normalized = " ".join(_normalize(readme_text).split())
    for phrase in (
        "see lkw",
        "run lkw locally",
        "why intergrax",
        "ai engineer / builder",
        "architect / principal engineer",
        "cto / engineering leader",
        "technical reviewer",
        "investor / strategic evaluator",
        "design partner / integrator",
        "primary daily-use conversational interface",
        "slack dm ask path is live-verified",
        "broader slack-first daily-use experience remains under productization",
    ):
        assert phrase in normalized, f"README missing routing marker: {phrase!r}"
    assert "Local Knowledge Workspace" in readme_text
    assert "Intergrax helps teams build" in readme_text
    assert "Try LKW" in readme_text
    assert "Primary Product Proof" in readme_text
    assert "Backend Product Alpha /" in readme_text
    assert "Featured platform-capability proof" in readme_text


def test_readme_multiplayer_positioning(readme_text: str) -> None:
    normalized = " ".join(_normalize(readme_text).split())
    for phrase in (
        "multiplayer ai",
        "architecture / roadmap stage",
        "runtime proof not yet established",
        "governed multi-principal collaboration",
        "not yet established",
    ):
        assert phrase in normalized, f"README missing Multiplayer marker: {phrase}"
    assert "featured platform-capability proof" in normalized
    assert "primary product proof" in normalized
    forbidden_positive = (
        "production multiplayer",
        "complete multi-agent collaboration",
        "proven a2a",
        "industry-leading multiplayer",
    )
    for phrase in forbidden_positive:
        assert phrase not in normalized, f"README contains hype claim: {phrase}"


def test_multiplayer_public_projection_links() -> None:
    assert MULTIPLAYER_ARCH_PATH.is_file()
    readme_text = _read(README_PATH)
    arch_text = _read(ARCHITECTURE_OVERVIEW_PATH)
    roadmap_text = _read(ROADMAP_PATH)
    hub_text = _read(HUB_PATH)

    assert "docs/project/capabilities/architecture/MULTIPLAYER_AI.md" in readme_text
    assert "../capabilities/architecture/MULTIPLAYER_AI.md" in arch_text
    assert "../capabilities/architecture/MULTIPLAYER_AI.md" in roadmap_text
    assert "capabilities/architecture/MULTIPLAYER_AI.md" in hub_text

    for doc_path, target in (
        (README_PATH, "docs/project/capabilities/architecture/MULTIPLAYER_AI.md"),
        (ARCHITECTURE_OVERVIEW_PATH, "../capabilities/architecture/MULTIPLAYER_AI.md"),
        (ROADMAP_PATH, "../capabilities/architecture/MULTIPLAYER_AI.md"),
        (HUB_PATH, "capabilities/architecture/MULTIPLAYER_AI.md"),
    ):
        resolved = (doc_path.parent / target.split("#", 1)[0]).resolve()
        assert resolved == MULTIPLAYER_ARCH_PATH.resolve(), (
            f"{doc_path.name} Multiplayer link does not resolve to canonical architecture"
        )


def test_readme_platform_extensibility_positioning(readme_text: str) -> None:
    normalized = " ".join(_normalize(readme_text).split())
    for phrase in (
        "platform extensibility",
        "governed extension/package model",
        "domain-owned contracts",
        "canonical architecture frozen",
        "implementation stages planned",
        "complete third-party install-to-runtime e2e proof not yet established",
    ):
        assert phrase in normalized, f"README missing Platform Extensibility marker: {phrase}"
    assert "primary product proof" in normalized
    assert "featured platform-capability proof" in normalized
    assert "partial" in normalized
    forbidden_positive = (
        "complete plugin ecosystem",
        "production-ready plugin platform",
        "fully unified plugin framework",
        "secure third-party plugins",
        "production-qualified plugin marketplace",
        "marketplace ready",
        "intergrax has no plugin system",
        "unified platform plugin framework",
        "already has a unified platform plugin",
    )
    for phrase in forbidden_positive:
        assert phrase not in normalized, f"README contains forbidden plugin claim: {phrase}"


def test_platform_extensibility_public_projection_links() -> None:
    assert PLATFORM_PLUGINS_ARCH_PATH.is_file()
    readme_text = _read(README_PATH)
    arch_text = _read(ARCHITECTURE_OVERVIEW_PATH)
    roadmap_text = _read(ROADMAP_PATH)
    hub_text = _read(HUB_PATH)

    assert "docs/project/architecture/PLATFORM_PLUGINS.md" in readme_text
    assert "PLATFORM_PLUGINS.md" in arch_text
    assert "../architecture/PLATFORM_PLUGINS.md" in roadmap_text
    assert "architecture/PLATFORM_PLUGINS.md" in hub_text

    for doc_path, target in (
        (README_PATH, "docs/project/architecture/PLATFORM_PLUGINS.md"),
        (ARCHITECTURE_OVERVIEW_PATH, "PLATFORM_PLUGINS.md"),
        (ROADMAP_PATH, "../architecture/PLATFORM_PLUGINS.md"),
        (HUB_PATH, "architecture/PLATFORM_PLUGINS.md"),
    ):
        resolved = (doc_path.parent / target.split("#", 1)[0]).resolve()
        assert resolved == PLATFORM_PLUGINS_ARCH_PATH.resolve(), (
            f"{doc_path.name} Platform Plugins link does not resolve to canonical architecture"
        )


def test_architecture_platform_extensibility_section(arch_text: str) -> None:
    normalized = " ".join(_normalize(arch_text).split())
    assert "platform extensibility as a strategic platform direction" in normalized
    for phrase in (
        "independent plugin package",
        "platform coordination",
        "domain capability contract",
        "host configuration / di",
        "governed intergrax execution",
        "not a universal platformplugin.execute",
        "not proof that every extension surface is already harmonized",
        "conceptual target architecture",
        "not proof that the full platform-level plugin lifecycle is implemented",
    ):
        assert phrase in normalized, f"ARCHITECTURE missing Platform Extensibility marker: {phrase}"
    assert "primary product proof" in normalized
    assert "featured platform-capability proof" in normalized
    assert "partial" in normalized


def test_roadmap_platform_extensibility_supporting_work() -> None:
    text = _read(ROADMAP_PATH)
    normalized = " ".join(_normalize(text).split())
    for phrase in (
        "platform extensibility / plugins",
        "strategic platform capability",
        "canonical cross-cutting architecture is frozen",
        "implementation stages",
        "not yet established",
        "token optimization",
        "featured platform-capability proof",
        "partial",
        "multiplayer ai",
    ):
        assert phrase in normalized, f"ROADMAP missing Platform Extensibility marker: {phrase}"
    assert "platform-plugin-" not in normalized


def test_readme_architecture_is_responsibility_model(readme_text: str) -> None:
    normalized = " ".join(_normalize(readme_text).split())
    for phrase in (
        "specialized product application",
        "application operating layer",
        "model / agent",
        "knowledge / tools / integrations / models",
        "evidence / provenance",
        "not a mandatory execution sequence",
        "configured resources",
    ):
        assert phrase in normalized, f"README missing architecture marker: {phrase}"

    stale_pipeline_fragments = (
        "applications, orchestration, agents, and the harness each own a clear slice",
        "app --> n[orchestration]",
        "n --> h[governed execution]",
        "h --> k[knowledge and memory]",
    )
    for fragment in stale_pipeline_fragments:
        assert fragment not in normalized, f"README contains stale pipeline: {fragment}"


def test_project_documentation_hub_routing() -> None:
    assert HUB_PATH.is_file()
    text = _read(HUB_PATH)
    for route in (
        "Understand Intergrax",
        "Try LKW",
        "Review proof",
        "Evaluate",
        "Build with Intergrax",
        "Review architecture",
        "applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md",
        "applications/local_workspace_application/docs/product/QUICKSTART.md",
        "proofs/PROOFS.md",
        "applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md",
        "builders/EVALUATION_GUIDE.md",
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
    assert (
        "[Builder Quick Start](builders/BUILDER_QUICKSTART.md) →\n"
        "  [Build With Intergrax](builders/BUILD_WITH_INTERGRAX.md)."
    ) in text
    assert "or the\n  [Evaluation Guide]" not in text


def test_public_map_reader_route_ownership_and_primary_intents() -> None:
    text = _read(PUBLIC_MAP_PATH)
    normalized = " ".join(_normalize(text).split())
    for phrase in (
        "product tour",
        "quick start",
        "proofs",
        "evaluation guide",
        "builder quick start",
        "build with intergrax",
        "architecture overview",
        "use cases",
        "partners",
        "technical documentation map",
    ):
        assert phrase in normalized, f"PUBLIC_MAP missing route: {phrase}"

    assert "deeper application composition planning" in normalized
    assert (
        "bounded evaluation method for one selected claim/workflow using a pinned revision"
        in normalized
    )
    assert "current evidence status / public evidence dashboard" in normalized
    assert (
        "outcome-gated direction across repeatability, complete intended outcome"
        in normalized
    )
    assert "planned validation" not in normalized
    assert "now, next and later" not in normalized
    assert "5–60" not in text
    assert not _INTERNAL_TASK_PATTERN.search(text)

    primary = _MERMAID_FENCE.findall(text)[0]
    for intent in (
        "Try the primary product",
        "Understand the product",
        "Check workflow fit",
        "Check current evidence",
        "Evaluate one claim",
        "Build",
        "Review architecture",
        "Prepare a pilot",
        "Deep technical review",
    ):
        assert intent in primary, f"PUBLIC_MAP Mermaid missing intent: {intent}"
    for forbidden in ("classDef", "style", "%%{init", "theme", "http://", "https://"):
        assert forbidden not in primary
    assert "B --> E" not in primary
    assert "Permission" not in primary
    assert "PROOFS" in primary


def test_public_map_separates_current_proof_paths_from_strategic_directions() -> None:
    text = _read(PUBLIC_MAP_PATH)
    normalized = " ".join(_normalize(text).split())
    current_section = text.split("## Current product / proof paths", 1)[1].split(
        "## Strategic directions", 1
    )[0]
    strategic_section = text.split("## Strategic directions", 1)[1].split(
        "## Public documents", 1
    )[0]
    current_norm = " ".join(_normalize(current_section).split())
    strategic_norm = " ".join(_normalize(strategic_section).split())

    assert "featured proof paths" not in normalized
    assert "local knowledge workspace" in current_norm
    assert "token optimization engine" in current_norm
    assert "agent marketplace" not in current_norm
    assert "multiplayer ai" in strategic_norm
    assert "platform extensibility / plugins" in strategic_norm
    assert "agent marketplace" in strategic_norm
    assert "future ecosystem direction" in strategic_norm
    assert "not a shipped public marketplace" in strategic_norm


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
        "python -m intergrax.scaffold new-stack",
        "uv run pytest applications/",
        "domain job",
    ):
        assert phrase in normalized, f"BUILDER_QUICKSTART missing semantic marker: {phrase}"

    assert "not mandatory" in normalized
    assert "not to begin builder onboarding" in normalized
    assert "no generic project scaffold" in normalized
    assert "universal application template" in normalized
    assert "stable universal public sdk" in normalized
    assert "uv sync --extra dev" not in builder_quickstart_text
    assert "uv run intergrax doctor" not in builder_quickstart_text
    assert "uv run pytest -m gate -q" not in builder_quickstart_text
    assert "repository baseline from the evaluation guide" not in normalized
    assert "you do not need to complete the evaluation guide first" in normalized
    assert "broader repository-level evaluation" in normalized


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


def test_why_business_strategic_thesis(why_text: str) -> None:
    normalized = _normalize(why_text)
    for phrase in (
        "business and strategic thesis",
        "duplication and fragmentation",
        "potential adopter or sponsor profiles",
        "lkw is the current product path used to test this thesis",
        "compounding value hypothesis",
        "commercialization gates",
        "real-user validation",
        "commercial validation",
    ):
        assert phrase in normalized, f"WHY missing business thesis marker: {phrase}"

    forbidden_promotional = (
        "tam",
        "market share",
        "revenue",
        "traction",
        "proven at scale",
        "enterprise-ready",
        "production-ready",
        "guaranteed savings",
        "roi",
    )
    for phrase in forbidden_promotional:
        assert phrase not in normalized, f"WHY contains promotional claim phrase: {phrase!r}"


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


def test_use_cases_business_evaluation_framing() -> None:
    normalized = " ".join(_normalize(_read(USE_CASES_PATH)).split())
    for phrase in (
        "who:",
        "current approach:",
        "pain:",
        "desired outcome:",
        "success signal:",
        "validation gap:",
        "evaluation question",
    ):
        assert phrase in normalized, f"USE_CASES missing business evaluation marker: {phrase}"

    for phrase in (
        "strongest current fit",
        "bounded technical fit",
        "not yet proven",
        "not a fit",
        "real-user validation and commercial validation are incomplete",
    ):
        assert phrase in normalized, f"USE_CASES missing fit taxonomy marker: {phrase}"


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


_LKW_PLATFORM_PROOF_PATH = (
    REPO_ROOT / "applications" / "local_workspace_application" / "docs" / "proof" / "LKW_PLATFORM_PROOF.md"
)


def test_quickstart_canonical_model_configuration_contract() -> None:
    text = _read(LKW_QUICKSTART_PATH)
    for key in FORBIDDEN_GENERATION_MODEL_ENV_NAMES:
        assert key not in text
    for var in (
        "INTERGRAX_LLM_PROVIDER",
        "INTERGRAX_LLM_MODEL",
        "INTERGRAX_EMBEDDING_PROVIDER",
        "INTERGRAX_EMBEDDING_MODEL",
    ):
        assert var in text, f"Quick Start missing canonical variable: {var}"
    assert "PLATFORM_CONFIGURATION.md" in text


def test_proofs_trusted_ask_public_entry() -> None:
    text = _read(PROOFS_PATH)
    normalized = " ".join(_normalize(text).split())
    assert "trusted ask" in normalized
    assert "lkw-ask-workspace-live" in normalized
    assert "trusted-ask-workspace-mvp-2" in text.lower()
    assert "no mixed indexed + authorized-live hybrid ask" in normalized


def test_readme_distinguishes_quickstart_from_hybrid_ask_certification(
    readme_text: str,
) -> None:
    normalized = " ".join(_normalize(readme_text).split())
    assert "indexed ask v1" in normalized
    assert "not hybrid ask certification" in normalized
    assert "lkw-hybrid-ask-indexed" in normalized


def test_product_tour_avoids_production_readiness_ambiguity() -> None:
    tour = " ".join(_normalize(_read(LKW_PRODUCT_TOUR_PATH)).split())
    assert "bounded production indexed ask path" not in tour
    assert "actual application/runtime indexed ask path" in tour


def test_proofs_public_text_avoids_public_evidence_eligible_reader_term() -> None:
    text = _read(PROOFS_PATH)
    assert "public_evidence_eligible" not in text


def test_lkw_platform_proof_core_outcome_first_summary() -> None:
    text = _read(_LKW_PLATFORM_PROOF_PATH)
    core_start = text.index("## Core Platform Proof")
    steps_start = text.index("1. LKW starts as a real Intergrax application.", core_start)
    summary = text[core_start:steps_start]
    assert "outcome-first" in summary.lower()
    assert "application starts and reports readiness" in summary.lower()
    assert "watched knowledge reaches the searchable index automatically" in summary.lower()


def test_public_reader_correction_wave_maturity_disclaimers_preserved(
    readme_text: str,
) -> None:
    for path in (README_PATH, PROOFS_PATH, LKW_PRODUCT_TOUR_PATH):
        normalized = " ".join(_normalize(_read(path)).split())
        for phrase in ("real-user validation", "commercial validation"):
            assert phrase in normalized, f"{path.name} missing disclaimer: {phrase}"
    readme_norm = " ".join(_normalize(readme_text).split())
    assert "backend product alpha" in readme_norm
    assert "partial" in readme_norm
    quickstart_norm = " ".join(_normalize(_read(LKW_QUICKSTART_PATH)).split())
    assert "commercial validation" in quickstart_norm


def test_lkw_quickstart_progressive_disclosure_structure() -> None:
    text = _read(LKW_QUICKSTART_PATH)
    before_run = text.index("## Before you run")
    run_cmd = text.index("## 1. Run one command")
    success = text.index("## 2. Success looks like this")
    what_happened = text.index("## 3. What just happened")
    what_proves = text.index("## What this proves")
    current_boundary = text.index("## Current boundary")
    config = text.index("## Configuration and first-run downloads")

    assert before_run < run_cmd < success < what_happened < what_proves < current_boundary < config
    assert text.index("AURORA-17") < config
    assert "### Verified Quick Start" in text
    assert "lkw_product_quickstart.txt" in text
    assert "persisted_run_verified=true" in text


def test_lkw_product_tour_presentation_contract() -> None:
    text = _read(LKW_PRODUCT_TOUR_PATH)
    normalized = " ".join(_normalize(text).split())

    assert "<picture>" in text
    assert "lkw-grounded-result-light.svg" in text
    assert "lkw-grounded-result-dark.svg" in text
    assert "## The LKW experience" in text
    assert "## What this proves" in text
    assert "## Current boundary" in text
    assert "not a screenshot of a finished application ui" in normalized
    assert "[LKW Quick Start](QUICKSTART.md)" in text
    assert normalized.index("## the lkw experience") < normalized.index("## current boundary")
