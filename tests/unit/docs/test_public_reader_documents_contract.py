# © Artur Czarnecki. All rights reserved.

"""PUBLIC-DOCS-COMMERCIALIZATION-6: core reader document contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "README.md"
WHY_PATH = REPO_ROOT / "WHY_INTERGRAX.md"
ARCHITECTURE_OVERVIEW_PATH = REPO_ROOT / "ARCHITECTURE_OVERVIEW.md"
BUILD_PATH = REPO_ROOT / "BUILD_WITH_INTERGRAX.md"
PUBLIC_MAP_PATH = REPO_ROOT / "docs" / "PUBLIC_DOCUMENTATION_MAP.md"
PUBLIC_ARCHITECTURE_PATH = REPO_ROOT / "docs" / "public-adoption" / "PUBLIC_DOCUMENTATION_ARCHITECTURE.md"
HERO_LIGHT_PATH = REPO_ROOT / "docs" / "assets" / "public" / "intergrax-hero-light.svg"
HERO_DARK_PATH = REPO_ROOT / "docs" / "assets" / "public" / "intergrax-hero-dark.svg"

_LEGAL_HEADER = (
    "<!--\n"
    "© Artur Czarnecki. All rights reserved.\n"
    "Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.\n"
    "See LICENSE for permitted evaluation, collaboration, and contribution use.\n"
    "-->"
)

_PRIMARY_SENTENCE = (
    "Intergrax helps teams build specialized agent applications without "
    "rebuilding the same policy, knowledge, evidence, integration, and "
    "execution foundations for every product."
)

_READER_PATHS = (WHY_PATH, ARCHITECTURE_OVERVIEW_PATH, BUILD_PATH)

_ARCH_OPENING = (
    "A concise view of how Intergrax separates product workflows, orchestration, "
    "agent decisions, governed execution, and evidence."
)

_BUILD_OPENING = (
    "Choose the right path to evaluate Intergrax, inspect its proof, "
    "or begin building a specialized application."
)

_FORBIDDEN_MAINTAINER_PHRASES = (
    "Freeze these concepts for public readers",
    "without copying detailed execution guides",
)

_INSTALL_SEQUENCE = (
    "git clone https://github.com/jakbuczarnecki/intergrax.git",
    "cd intergrax",
    "uv sync --extra dev",
    "uv run intergrax doctor",
    "uv run pytest -m gate -q",
)

_FORBIDDEN_INSTALL_CHAINS = (
    "git clone https://github.com/jakbuczarnecki/intergrax.git && cd intergrax",
    "uv sync --extra dev && uv run intergrax doctor",
)

_EVIDENCE_LIMITATIONS = (
    "production runtime certification",
    "security/compliance attestation",
    "real provider execution",
    "real LLM evaluation",
    "billing",
    "provider pricing",
    "cloud cost estimation",
    "product-specific acceptance",
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
    "production-readiness",
    "responsibility boundaries",
)

_BOUNDARY_PHRASES_BUILD = (
    "active r&d",
    "bounded",
    "production permission",
    "license",
)

_LINK_TARGETS_BY_DOC: dict[Path, tuple[str, ...]] = {
    WHY_PATH: (
        "PROOFS.md",
        "docs/public-adoption/LKW_PLATFORM_PROOF.md",
        "docs/features/token_optimization/README.md",
        "ARCHITECTURE_OVERVIEW.md",
        "BUILD_WITH_INTERGRAX.md",
    ),
    ARCHITECTURE_OVERVIEW_PATH: (
        "PROOFS.md",
        "docs/public-adoption/LKW_PLATFORM_PROOF.md",
        "docs/features/token_optimization/README.md",
        "docs/public-adoption/TOKEN_OPTIMIZATION_CLAIMS.md",
        "docs/DOCUMENTATION_MAP.md",
        "docs/guides/INTERGRAX_HARNESS_NARRATIVE.md",
    ),
    BUILD_PATH: (
        "PROOFS.md",
        "docs/public-adoption/LKW_PLATFORM_PROOF.md",
        "docs/features/token_optimization/README.md",
        "docs/public-adoption/TOKEN_OPTIMIZATION_CLAIMS.md",
        "docs/PUBLIC_DOCUMENTATION_MAP.md",
        "EVALUATION_GUIDE.md",
        "COLLABORATION.md",
        "LICENSE",
        "README.md#quick-start",
        "docs/guides/AGENT_CREATION_GUIDE.md",
        "applications/USAGE.md",
        "ARCHITECTURE_OVERVIEW.md",
        "docs/DOCUMENTATION_MAP.md",
    ),
}

_LINK_CHECK_PATHS = (
    WHY_PATH,
    ARCHITECTURE_OVERVIEW_PATH,
    BUILD_PATH,
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
    docs = {"WHY": why_text, "ARCHITECTURE": arch_text, "BUILD": build_text}
    forbidden_tokens = ("classDef", "style", "%%{init", "theme", "http://", "https://")
    for name, text in docs.items():
        blocks = _MERMAID_FENCE.findall(text)
        assert len(blocks) >= 1, f"{name} must contain at least one Mermaid block"
        for block in blocks:
            for token in forbidden_tokens:
                assert token not in block, f"{name}: forbidden Mermaid token {token!r}"


def test_hero_reuse(arch_text: str) -> None:
    assert "docs/assets/public/intergrax-hero-light.svg" in arch_text
    assert "docs/assets/public/intergrax-hero-dark.svg" in arch_text
    assert "<picture>" in arch_text
    assert 'alt="Intergrax connects specialized agent applications' in arch_text
    assert HERO_LIGHT_PATH.is_file()
    assert HERO_DARK_PATH.is_file()
    assert "intergrax-hero" not in arch_text.replace("intergrax-hero-light.svg", "").replace(
        "intergrax-hero-dark.svg", ""
    )


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
    for path in _READER_PATHS:
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
    for path in _READER_PATHS:
        text = _read(path)
        match = _INTERNAL_TASK_PATTERN.search(text)
        assert match is None, f"{path.name} contains internal task ID: {match.group()}"


def test_readme_routing(readme_text: str) -> None:
    for link in (
        "WHY_INTERGRAX.md",
        "ARCHITECTURE_OVERVIEW.md",
        "BUILD_WITH_INTERGRAX.md",
    ):
        assert link in readme_text, f"README missing link: {link}"
    assert _PRIMARY_SENTENCE in readme_text


def test_public_map_synchronization() -> None:
    text = _read(PUBLIC_MAP_PATH)
    for doc in ("WHY_INTERGRAX.md", "ARCHITECTURE_OVERVIEW.md", "BUILD_WITH_INTERGRAX.md"):
        assert doc in text
    planned_section = text.split("## Planned public structure", 1)[1]
    for doc in ("WHY_INTERGRAX.md", "ARCHITECTURE_OVERVIEW.md", "BUILD_WITH_INTERGRAX.md"):
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


def test_readme_install_sequence(readme_text: str) -> None:
    for chain in _FORBIDDEN_INSTALL_CHAINS:
        assert chain not in readme_text, f"README chains install commands: {chain!r}"

    lines = readme_text.splitlines()
    for index, command in enumerate(_INSTALL_SEQUENCE):
        assert command in lines, f"README missing install line: {command!r}"
        if index > 0:
            prev_command = _INSTALL_SEQUENCE[index - 1]
            prev_idx = lines.index(prev_command)
            curr_idx = lines.index(command)
            assert curr_idx == prev_idx + 1, (
                f"Install commands not consecutive: {prev_command!r} then {command!r}"
            )


def test_evidence_limitations_bulleted(readme_text: str) -> None:
    for limitation in _EVIDENCE_LIMITATIONS:
        assert f"- {limitation}" in readme_text, (
            f"README missing evidence limitation bullet: {limitation!r}"
        )


def test_brevity() -> None:
    limits = {
        WHY_PATH: 240,
        ARCHITECTURE_OVERVIEW_PATH: 280,
        BUILD_PATH: 300,
        README_PATH: 300,
    }
    for path, max_lines in limits.items():
        count = len(_read(path).splitlines())
        assert count <= max_lines, f"{path.name} has {count} lines (max {max_lines})"


def test_hero_assets_exist() -> None:
    assert HERO_LIGHT_PATH.is_file()
    assert HERO_DARK_PATH.is_file()


def test_why_canonical_sentence(why_text: str) -> None:
    assert _PRIMARY_SENTENCE in why_text
