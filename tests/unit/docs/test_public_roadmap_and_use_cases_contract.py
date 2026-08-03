# © Artur Czarnecki. All rights reserved.

"""PUBLIC-DOCS-COMMERCIALIZATION-7: roadmap and use-cases public document contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "README.md"
ROADMAP_PATH = REPO_ROOT / "ROADMAP.md"
USE_CASES_PATH = REPO_ROOT / "USE_CASES.md"
PROOFS_PATH = REPO_ROOT / "PROOFS.md"
PUBLIC_MAP_PATH = REPO_ROOT / "docs" / "PUBLIC_DOCUMENTATION_MAP.md"
PUBLIC_ARCHITECTURE_PATH = REPO_ROOT / "docs" / "public-adoption" / "PUBLIC_DOCUMENTATION_ARCHITECTURE.md"

_LEGAL_HEADER = (
    "<!--\n"
    "© Artur Czarnecki. All rights reserved.\n"
    "Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.\n"
    "See LICENSE for permitted evaluation, collaboration, and contribution use.\n"
    "-->"
)

_ROADMAP_H1 = "# Intergrax Public Roadmap"
_USE_CASES_H1 = "# Intergrax Use Cases"

_ROADMAP_PHASE_HEADINGS = (
    "## Now — Make LKW repeatable",
    "## Next — Validate the complete knowledge workflow",
    "## Later — Expand from evidence",
)

_READER_DOCS = (ROADMAP_PATH, USE_CASES_PATH)

_MERMAID_FENCE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

_INTERNAL_TASK_PATTERN = re.compile(
    r"(CTX-UCL-|TOKEN-10|LKW-[A-Z0-9]|GOOGLE-WORKSPACE-|MSGRAPH-|SLACK-KNOWLEDGE-|"
    r"PUBLIC-DOCS-COMMERCIALIZATION-)",
    re.IGNORECASE,
)

_FORBIDDEN_ARCH_PHRASES = (
    "Agent OS",
    "Nexus",
    "Tier-0",
    "Tier-1",
    "Tier-2",
    "Tier-3",
    "The harness is the product",
)

_FORBIDDEN_STATUS_TERMS = (
    "READY_FOR_REVIEW",
    "CHANGES_REQUIRED",
    "ACCEPTED",
    "DONE",
    "IN_PROGRESS",
)

_FORBIDDEN_CLAIM_PHRASES = (
    "finished saas",
    "production ready",
    "commercially validated",
    "universal token reduction",
    "production-proven savings",
)

_NEGATION_MARKERS = (
    "not ",
    "does not",
    "do not",
    "no ",
    "incomplete",
    "remain",
    "without",
    "not a fit",
    "not claimed",
    "not complete",
    "another approach",
)

_PERCENT_PATTERN = re.compile(r"\d+\s*%")

_RELEASE_DATE_PATTERNS = (
    re.compile(r"Q[1-4]\s+20\d{2}", re.IGNORECASE),
    re.compile(r"will ship by", re.IGNORECASE),
    re.compile(r"will be complete by", re.IGNORECASE),
    re.compile(r"release date:", re.IGNORECASE),
)

_LINK_CHECK_PATHS = (
    ROADMAP_PATH,
    USE_CASES_PATH,
    README_PATH,
    PUBLIC_MAP_PATH,
)

_REQUIRED_LINKS_ROADMAP = (
    "PROOFS.md",
    "docs/public-adoption/LKW_PLATFORM_PROOF.md",
    "USE_CASES.md",
    "BUILD_WITH_INTERGRAX.md",
    "docs/PUBLIC_DOCUMENTATION_MAP.md",
)

_REQUIRED_LINKS_USE_CASES = (
    "PROOFS.md",
    "docs/public-adoption/LKW_PLATFORM_PROOF.md",
    "docs/features/token_optimization/README.md",
    "BUILD_WITH_INTERGRAX.md",
    "EVALUATION_GUIDE.md",
    "PARTNERS.md",
    "COLLABORATION.md",
    "LICENSE",
    "ROADMAP.md",
    "docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md",
)

_README_PRESERVED_LINKS = (
    "WHY_INTERGRAX.md",
    "ARCHITECTURE_OVERVIEW.md",
    "BUILD_WITH_INTERGRAX.md",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalize(text: str) -> str:
    return re.sub(r"[*_`]", "", text).lower()


def _before_first_h3(text: str) -> str:
    match = re.search(r"^### ", text, re.MULTILINE)
    if match:
        return text[: match.start()]
    return text


@pytest.fixture(scope="module")
def roadmap_text() -> str:
    return _read(ROADMAP_PATH)


@pytest.fixture(scope="module")
def use_cases_text() -> str:
    return _read(USE_CASES_PATH)


@pytest.fixture(scope="module")
def readme_text() -> str:
    return _read(README_PATH)


def test_files_exist_and_legal_headers() -> None:
    for path in _READER_DOCS:
        assert path.is_file(), f"Missing document: {path}"
        assert _read(path).startswith(_LEGAL_HEADER), f"Missing legal header in {path.name}"


def test_required_h1_titles(roadmap_text: str, use_cases_text: str) -> None:
    assert roadmap_text.splitlines()[6].strip() == _ROADMAP_H1
    assert use_cases_text.splitlines()[6].strip() == _USE_CASES_H1


def test_first_screen_contract(roadmap_text: str, use_cases_text: str) -> None:
    for text in (roadmap_text, use_cases_text):
        early = _before_first_h3(text)
        early_norm = _normalize(early)
        for phrase in ("source-available", "active r&d", "partial"):
            assert phrase in early_norm, f"Missing first-screen phrase {phrase!r}"

    roadmap_early = _normalize(_before_first_h3(roadmap_text))
    for phrase in (
        "outcome-gated",
        "not a release-date commitment",
        "real-user validation incomplete",
        "commercial validation incomplete",
    ):
        assert phrase in roadmap_early, f"ROADMAP missing boundary phrase: {phrase}"

    use_cases_early = _normalize(_before_first_h3(use_cases_text))
    assert "primary product proof" in use_cases_early
    assert "backend product alpha / mvp" in use_cases_early


def test_at_a_glance(roadmap_text: str, use_cases_text: str) -> None:
    for text in (roadmap_text, use_cases_text):
        assert "## At a glance" in text


def test_mermaid_blocks(roadmap_text: str, use_cases_text: str) -> None:
    forbidden_tokens = ("classDef", "style", "%%{init", "theme", "http://", "https://")
    for name, text in (("ROADMAP", roadmap_text), ("USE_CASES", use_cases_text)):
        blocks = _MERMAID_FENCE.findall(text)
        assert len(blocks) >= 1, f"{name} must contain at least one Mermaid block"
        for block in blocks:
            for token in forbidden_tokens:
                assert token not in block, f"{name}: forbidden Mermaid token {token!r}"


def test_roadmap_phase_headings(roadmap_text: str) -> None:
    for heading in _ROADMAP_PHASE_HEADINGS:
        assert heading in roadmap_text, f"ROADMAP missing phase heading: {heading}"


def test_outcome_gated_roadmap(roadmap_text: str) -> None:
    roadmap_norm = _normalize(roadmap_text)
    for phrase in (
        "user result",
        "proof required",
        "real-user validation",
        "evidence-driven expansion",
    ):
        assert phrase in roadmap_norm, f"ROADMAP missing outcome phrase: {phrase}"

    for term in _FORBIDDEN_STATUS_TERMS:
        assert term not in roadmap_text, f"ROADMAP leaks internal status: {term}"


def test_no_internal_task_ids(roadmap_text: str, use_cases_text: str) -> None:
    for path, text in ((ROADMAP_PATH, roadmap_text), (USE_CASES_PATH, use_cases_text)):
        assert not _INTERNAL_TASK_PATTERN.search(text), (
            f"{path.name} contains forbidden internal task ID pattern"
        )


def test_no_internal_architecture_language(roadmap_text: str, use_cases_text: str) -> None:
    for path, text in ((ROADMAP_PATH, roadmap_text), (USE_CASES_PATH, use_cases_text)):
        for phrase in _FORBIDDEN_ARCH_PHRASES:
            assert phrase not in text, f"{path.name} contains forbidden phrase: {phrase!r}"


def test_use_case_classifications(use_cases_text: str) -> None:
    required = (
        "Primary product proof",
        "Featured platform-capability proof",
        "Strongest current fit",
        "Reasonable technical evaluation",
        "Planned fit",
        "Not a fit today",
    )
    for phrase in required:
        assert phrase in use_cases_text, f"USE_CASES missing classification: {phrase}"


def test_claim_boundaries(roadmap_text: str, use_cases_text: str) -> None:
    combined = roadmap_text + use_cases_text
    combined_norm = _normalize(combined)
    assert "hybrid ask" in combined_norm and "not complete" in combined_norm
    assert "real-user validation incomplete" in combined_norm
    assert "commercial validation incomplete" in combined_norm
    assert "universal savings not claimed" in combined_norm or "universal savings are not claimed" in combined_norm

    for path, text in ((ROADMAP_PATH, roadmap_text), (USE_CASES_PATH, use_cases_text)):
        assert not _PERCENT_PATTERN.search(text), f"{path.name}: numeric savings percentage"
        lower = _normalize(text)
        for phrase in _FORBIDDEN_CLAIM_PHRASES:
            start = 0
            while True:
                idx = lower.find(phrase, start)
                if idx == -1:
                    break
                context = lower[max(0, idx - 60) : idx + len(phrase) + 60]
                assert any(marker in context for marker in _NEGATION_MARKERS), (
                    f"{path.name}: positive forbidden claim {phrase!r}"
                )
                start = idx + 1


def test_required_links(roadmap_text: str, use_cases_text: str) -> None:
    for target in _REQUIRED_LINKS_ROADMAP:
        assert target in roadmap_text, f"ROADMAP missing link: {target}"
    for target in _REQUIRED_LINKS_USE_CASES:
        assert target in use_cases_text, f"USE_CASES missing link: {target}"


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


def test_readme_routing(readme_text: str) -> None:
    assert "USE_CASES.md" in readme_text
    assert "ROADMAP.md" in readme_text
    for link in _README_PRESERVED_LINKS:
        assert link in readme_text, f"README missing preserved link: {link}"


def test_public_map_synchronization() -> None:
    text = _read(PUBLIC_MAP_PATH)
    assert "USE_CASES.md" in text or "Use Cases" in text
    assert "ROADMAP.md" in text or "Roadmap" in text
    assert "product-validation program is heading" in text.lower() or "outcome-gated" in text.lower()
    map_norm = _normalize(text)
    assert "use cases" in map_norm
    assert "roadmap" in map_norm
    assert "planned validation and not-fit" in map_norm or "outcome-gated" in map_norm


def test_architecture_synchronization() -> None:
    text = _read(PUBLIC_ARCHITECTURE_PATH)
    assert "PUBLIC-DOCS-COMMERCIALIZATION-7" in text
    assert "USE_CASES.md" in text
    assert "ROADMAP.md" in text
    assert "implemented / refreshed" in text or "implemented / refreshed" in text.lower()


def test_brevity() -> None:
    roadmap_lines = len(_read(ROADMAP_PATH).splitlines())
    use_cases_lines = len(_read(USE_CASES_PATH).splitlines())
    readme_lines = len(_read(README_PATH).splitlines())
    assert roadmap_lines <= 240, f"ROADMAP too long: {roadmap_lines} lines"
    assert use_cases_lines <= 280, f"USE_CASES too long: {use_cases_lines} lines"
    assert readme_lines <= 300, f"README too long: {readme_lines} lines"


def test_no_release_date_promises(roadmap_text: str, use_cases_text: str) -> None:
    for path, text in ((ROADMAP_PATH, roadmap_text), (USE_CASES_PATH, use_cases_text)):
        for pattern in _RELEASE_DATE_PATTERNS:
            assert not pattern.search(text), (
                f"{path.name}: release-date promise pattern {pattern.pattern!r}"
            )
