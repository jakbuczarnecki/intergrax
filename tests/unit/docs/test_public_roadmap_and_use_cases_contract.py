# © Artur Czarnecki. All rights reserved.

"""PUBLIC-DOCS-COMMERCIALIZATION-7: roadmap and use-cases public document contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "README.md"
ROADMAP_PATH = REPO_ROOT / "docs" / "project" / "overview" / "ROADMAP.md"
USE_CASES_PATH = REPO_ROOT / "docs" / "project" / "overview" / "USE_CASES.md"
PROOFS_PATH = REPO_ROOT / "docs" / "project" / "proofs" / "PROOFS.md"
PUBLIC_MAP_PATH = REPO_ROOT / "docs" / "project" / "community" / "PUBLIC_DOCUMENTATION_MAP.md"
PUBLIC_ARCHITECTURE_PATH = REPO_ROOT / "docs" / "project" / "maintainers" / "public-adoption" / "PUBLIC_DOCUMENTATION_ARCHITECTURE.md"

_LEGAL_HEADER = (
    "<!--\n"
    "© Artur Czarnecki. All rights reserved.\n"
    "Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.\n"
    "See LICENSE for permitted evaluation, collaboration, and contribution use.\n"
    "-->"
)

_ROADMAP_H1 = "# Intergrax Public Roadmap"
_USE_CASES_H1 = "# Intergrax Use Cases"

_ROADMAP_STAGE_MARKERS = (
    ("## now", "make the primary workflow repeatable"),
    ("## next", "prove the complete intended knowledge outcome"),
    ("## validate", "establish real-user value and repeat use"),
    ("## expand", "evidence-driven expansion"),
    ("## harden / package", "improve operations after validated use"),
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

_FORBIDDEN_ROADMAP_PROVIDER_SEQUENCE_PHRASES = (
    "durable slack dm interaction",
    "complete connected slack knowledge workflow",
    "first governed google workspace lkw proof",
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
    "../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md",
    "USE_CASES.md",
    "BUILD_WITH_INTERGRAX.md",
    "../community/PUBLIC_DOCUMENTATION_MAP.md",
)

_REQUIRED_LINKS_USE_CASES = (
    "PROOFS.md",
    "../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md",
    "../capabilities/token_optimization/README.md",
    "BUILD_WITH_INTERGRAX.md",
    "EVALUATION_GUIDE.md",
    "PARTNERS.md",
    "COLLABORATION.md",
    "LICENSE",
    "ROADMAP.md",
    "case-studies/BOUNDARYATTEST_ATTESTATION_POC.md",
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


def _through_at_a_glance(text: str) -> str:
    """Return document start through complete ``## At a glance`` section."""
    at_glance = re.search(r"^## At a glance\s*$", text, re.MULTILINE)
    if not at_glance:
        raise AssertionError("Missing ## At a glance section")
    after_at_glance = text[at_glance.end() :]
    next_h2 = re.search(r"^## ", after_at_glance, re.MULTILINE)
    if not next_h2:
        raise AssertionError("Missing H2 section after ## At a glance")
    return text[: at_glance.end() + next_h2.start()]


def _h2_section(text: str, heading: str) -> str:
    """Extract content from an H2 heading through the next H2 or EOF."""
    pattern = re.compile(rf"^{re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise AssertionError(f"Missing section: {heading}")
    after = text[match.end() :]
    next_h2 = re.search(r"^## ", after, re.MULTILINE)
    end = match.end() + (next_h2.start() if next_h2 else len(after))
    return text[match.start() : end]


def _fit_matrix_section(text: str) -> str:
    return _h2_section(text, "## Fit matrix")


_READER_HYBRID_BOUNDARY = (
    "Indexed Ask through production Hybrid Ask is boundedly demonstrated; "
    "authorized live evidence combined with indexed evidence is not yet established."
)

_FORBIDDEN_READER_MAINTAINER_PHRASES = (
    "Do not claim",
    "must not claim",
    "maintainer",
    "public wording",
)

_ROADMAP_NEGATIVE_BULLETS = (
    "No finished hosted SaaS.",
    "No claim that mixed indexed + authorized live Hybrid Ask is complete.",
    "No claim of complete live-provider access or a complete provider catalog.",
    "No completed real-user validation.",
    "No completed commercial validation.",
    "No claim of universal production readiness.",
    "No universal token-savings claim.",
    "No fixed release-date commitment.",
)

_USE_CASES_NEGATIVE_BULLETS = (
    "No finished hosted SaaS",
    "No complete Hybrid Ask combining indexed and authorized live evidence",
    "No complete multi-provider live access",
    "No universal production certification",
    "No compliance certification",
    "No unrestricted open-source rights",
    "No universal token or cost reduction",
    "No automatic acceptance of every proposed use case",
)

_PUBLIC_MAP_FORBIDDEN_TIER_TERMS = (
    "Tier-0",
    "Tier-1",
    "Tier-2",
    "Tier-3",
)


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
        early = _through_at_a_glance(text)
        early_norm = _normalize(early)
        for phrase in ("source-available", "active r&d", "partial"):
            assert phrase in early_norm, f"Missing first-screen phrase {phrase!r}"

    roadmap_early = _normalize(_through_at_a_glance(roadmap_text))
    for phrase in (
        "outcome-gated",
        "not a release-date commitment",
        "real-user validation incomplete",
        "commercial validation incomplete",
        "primary product focus",
        "local knowledge workspace (lkw)",
        "backend product alpha / mvp",
        "partial",
    ):
        assert phrase in roadmap_early, f"ROADMAP missing boundary phrase: {phrase}"
    for provider in ("slack", "google", "microsoft", "jira"):
        assert provider not in roadmap_early, f"ROADMAP first screen names provider: {provider}"

    use_cases_early = _normalize(_through_at_a_glance(use_cases_text))
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
    headings = [line.strip().lower() for line in roadmap_text.splitlines() if line.startswith("## ")]
    for heading_prefix, marker in _ROADMAP_STAGE_MARKERS:
        assert any(heading.startswith(heading_prefix) for heading in headings), (
            f"ROADMAP missing semantic phase heading: {heading_prefix}"
        )
        assert marker in _normalize(roadmap_text), (
            f"ROADMAP missing semantic phase marker: {marker}"
        )


def test_outcome_gated_roadmap(roadmap_text: str) -> None:
    roadmap_norm = _normalize(roadmap_text)
    for phrase in (
        "user / product outcome",
        "evidence required",
        "real-user validation",
        "evidence-driven expansion",
    ):
        assert phrase in roadmap_norm, f"ROADMAP missing outcome phrase: {phrase}"

    for term in _FORBIDDEN_STATUS_TERMS:
        assert term not in roadmap_text, f"ROADMAP leaks internal status: {term}"


def test_roadmap_does_not_track_provider_rollouts(roadmap_text: str) -> None:
    normalized = _normalize(roadmap_text)
    for phrase in _FORBIDDEN_ROADMAP_PROVIDER_SEQUENCE_PHRASES:
        assert phrase not in normalized, (
            f"ROADMAP contains provider rollout milestone: {phrase}"
        )


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
    fit_matrix = _fit_matrix_section(use_cases_text)
    normalized = _normalize(fit_matrix)
    for phrase in (
        "strongest current fit",
        "bounded technical fit",
        "not yet proven",
        "not a fit",
        "evaluation question",
    ):
        assert phrase in normalized, f"USE_CASES fit matrix missing class: {phrase}"

    assert "indexed knowledge combined with authorized live evidence" in normalized
    assert "mixed indexed + authorized live" in _normalize(use_cases_text)
    assert "remains incomplete" in _normalize(use_cases_text)
    assert "planned fit" not in normalized, (
        "USE_CASES must not use ambiguous Planned fit classification"
    )

    forbidden_provider_rows = (
        "durable connected slack knowledge workflow",
        "slack as interaction surface and approved knowledge source",
        "governed google workspace knowledge inside lkw",
        "first bounded google workspace proof",
        "provider rollout",
        "next product milestone",
    )
    for phrase in forbidden_provider_rows:
        assert phrase not in normalized, (
            f"USE_CASES fit matrix contains provider roadmap detail: {phrase}"
        )


def test_reader_facing_copy(use_cases_text: str) -> None:
    for phrase in _FORBIDDEN_READER_MAINTAINER_PHRASES:
        assert phrase not in use_cases_text, (
            f"USE_CASES contains maintainer-style phrase: {phrase!r}"
        )
    assert _READER_HYBRID_BOUNDARY in use_cases_text, (
        "USE_CASES missing reader-facing Hybrid Ask boundary sentence"
    )


def test_explicit_negative_lists(roadmap_text: str, use_cases_text: str) -> None:
    roadmap_negatives = _h2_section(roadmap_text, "## What is not promised")
    for bullet in _ROADMAP_NEGATIVE_BULLETS:
        assert bullet in roadmap_negatives, f"ROADMAP missing negative bullet: {bullet!r}"

    use_cases_negatives = _h2_section(
        use_cases_text, "## What Intergrax does not currently offer"
    )
    for bullet in _USE_CASES_NEGATIVE_BULLETS:
        assert bullet in use_cases_negatives, f"USE_CASES missing negative bullet: {bullet!r}"


def test_claim_boundaries(roadmap_text: str, use_cases_text: str) -> None:
    combined = roadmap_text + use_cases_text
    combined_norm = _normalize(combined)
    assert "hybrid ask" in combined_norm
    assert "not complete" in combined_norm or "remains incomplete" in combined_norm
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
    for term in _PUBLIC_MAP_FORBIDDEN_TIER_TERMS:
        assert term not in text, f"PUBLIC_DOCUMENTATION_MAP contains internal tier term: {term!r}"


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


def test_indexed_hybrid_ask_claim_boundary_across_public_docs(
    readme_text: str, use_cases_text: str, roadmap_text: str
) -> None:
    """Cross-document: indexed Hybrid Ask is discoverable; mixed indexed+live is not complete."""
    proofs_text = _read(PROOFS_PATH)
    combined = f"{readme_text}\n{proofs_text}\n{use_cases_text}\n{roadmap_text}"
    combined_norm = _normalize(combined)

    # 1. Indexed Hybrid Ask evidence is discoverable in the public claim surface.
    proofs_norm = _normalize(proofs_text)
    readme_norm = _normalize(readme_text)
    use_cases_norm = _normalize(use_cases_text)
    indexed_proven_in_proofs = any(
        marker in proofs_norm
        for marker in (
            "production hybrid ask indexed",
            "indexed-evidence path",
            "production indexed ask path through hybrid ask",
            "bounded indexed hybrid ask branch",
            "lkw-hybrid-ask-indexed",
        )
    )
    indexed_proven_in_readme = (
        "hybrid ask" in readme_norm
        and (
            "production code path" in readme_norm
            or "real application code path" in readme_norm
        )
        and "indexed" in readme_norm
        and ("proven indexed branch" in readme_norm or "proven scope" in readme_norm)
    )
    assert indexed_proven_in_proofs or indexed_proven_in_readme
    assert (
        "indexed ask through production hybrid ask" in use_cases_norm
        or "indexed path through production hybrid ask" in use_cases_norm
    )

    # 2. Complete indexed + live Hybrid Ask remains explicitly incomplete.
    mixed_markers = (
        "hybrid ask combining indexed and authorized live evidence",
        "hybrid ask combining indexed and live evidence",
        "authorized live evidence combined with indexed evidence",
    )
    assert any(marker in combined_norm for marker in mixed_markers)
    assert "not established" in _normalize(proofs_text) or "not yet established" in combined_norm
    assert "not complete" in combined_norm

    # 3. ROADMAP still treats combined indexed + live evidence as a future target.
    next_section = _normalize(
        _h2_section(roadmap_text, "## NEXT — Prove the complete intended knowledge outcome")
    )
    assert "bounded indexed ask path exists" in next_section
    assert "mixed indexed + authorized live hybrid ask remains incomplete" in next_section
    assert "complete live-provider access remains incomplete" in next_section

    # 4. No public document positively claims complete Hybrid Ask.
    # Negated forms such as "No claim that Hybrid Ask is complete" are allowed.
    for path, text in (
        (README_PATH, readme_text),
        (PROOFS_PATH, proofs_text),
        (USE_CASES_PATH, use_cases_text),
        (ROADMAP_PATH, roadmap_text),
    ):
        lower = _normalize(text)
        start = 0
        while True:
            idx = lower.find("hybrid ask is complete", start)
            if idx == -1:
                break
            context = lower[max(0, idx - 80) : idx + len("hybrid ask is complete") + 40]
            assert any(marker in context for marker in _NEGATION_MARKERS), (
                f"{path.name}: positive complete-Hybrid-Ask claim near {context!r}"
            )
            start = idx + 1
    # Blanket incomplete wording must not erase the proven indexed branch.
    assert "completed Hybrid Ask\n" not in proofs_text
    assert "- Hybrid Ask;\n" not in readme_text
