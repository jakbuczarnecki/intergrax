# © Artur Czarnecki. All rights reserved.

"""PUBLIC-DOCS-COMMERCIALIZATION-8: partner, collaboration and FAQ contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
PARTNERS_PATH = REPO_ROOT / "docs" / "project" / "community" / "PARTNERS.md"
COLLABORATION_PATH = REPO_ROOT / "docs" / "project" / "community" / "COLLABORATION.md"
FAQ_PATH = REPO_ROOT / "docs" / "project" / "overview" / "FAQ.md"
EVALUATION_GUIDE_PATH = REPO_ROOT / "docs" / "project" / "builders" / "EVALUATION_GUIDE.md"
BUILDER_QUICKSTART_PATH = REPO_ROOT / "docs" / "project" / "builders" / "BUILDER_QUICKSTART.md"
BUILD_WITH_PATH = REPO_ROOT / "docs" / "project" / "builders" / "BUILD_WITH_INTERGRAX.md"
PROOFS_PATH = REPO_ROOT / "docs" / "project" / "proofs" / "PROOFS.md"
LICENSE_PATH = REPO_ROOT / "LICENSE"
README_PATH = REPO_ROOT / "README.md"
PUBLIC_MAP_PATH = REPO_ROOT / "docs" / "project" / "community" / "PUBLIC_DOCUMENTATION_MAP.md"
PUBLIC_ARCHITECTURE_PATH = REPO_ROOT / "docs" / "project" / "maintainers" / "public-adoption" / "PUBLIC_DOCUMENTATION_ARCHITECTURE.md"
CONTRIBUTING_PATH = REPO_ROOT / "CONTRIBUTING.md"
SECURITY_PATH = REPO_ROOT / "SECURITY.md"
ISSUE_TEMPLATE_PATHS = tuple(
    REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / name
    for name in (
        "bug_report.yml",
        "config.yml",
        "design_partner_interest.yml",
        "integration_proposal.yml",
        "proof_path_feedback.yml",
        "scenario_proposal.yml",
    )
)

_COLLABORATION_URL = (
    "https://github.com/jakbuczarnecki/intergrax/blob/main/"
    "docs/project/community/COLLABORATION.md"
)
_ROADMAP_URL = (
    "https://github.com/jakbuczarnecki/intergrax/blob/main/"
    "docs/project/overview/ROADMAP.md"
)
_OLD_COLLABORATION_URL = (
    "https://github.com/jakbuczarnecki/intergrax/blob/main/COLLABORATION.md"
)
_OLD_ROADMAP_URL = (
    "https://github.com/jakbuczarnecki/intergrax/blob/main/ROADMAP.md"
)

_LEGAL_HEADER = (
    "<!--\n"
    "© Artur Czarnecki. All rights reserved.\n"
    "Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.\n"
    "See LICENSE for permitted evaluation, collaboration, and contribution use.\n"
    "-->"
)

_PARTNERS_H1 = "# Partners and Pilots"
_COLLABORATION_H1 = "# Collaborate with Intergrax"
_FAQ_H1 = "# Intergrax FAQ"

_READER_DOCS = (PARTNERS_PATH, COLLABORATION_PATH, FAQ_PATH)

_MERMAID_FENCE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

_INTERNAL_TASK_PATTERN = re.compile(
    r"(PUBLIC-DOCS-COMMERCIALIZATION-|CTX-UCL-|TOKEN-10|LKW-[A-Z0-9]|GOOGLE-WORKSPACE-|MSGRAPH-)",
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

_FORBIDDEN_PERMISSION_PHRASES = (
    "production use is permitted",
    "commercial use is permitted",
    "you may use Intergrax in production",
    "you may sell Intergrax",
    "you may host Intergrax as SaaS",
    "redistribution is permitted",
    "a pilot is automatically permitted",
    "contacting the maintainer grants permission",
)

_FORBIDDEN_FAQ_HEADINGS = (
    "## What is Nexus?",
    "## What do the tiers mean?",
    "## Can I create a competing product?",
)

_CONTACT_EMAIL = "jakbu.czarnecki.83@gmail.com"

_LINK_CHECK_PATHS = (
    PARTNERS_PATH,
    COLLABORATION_PATH,
    FAQ_PATH,
    PUBLIC_MAP_PATH,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalize(text: str) -> str:
    return re.sub(r"[*_`]", "", text).lower()


def _assert_semantic_markers(
    text: str, markers: tuple[str, ...], *, context: str = ""
) -> None:
    normalized = _normalize(text)
    for marker in markers:
        assert marker in normalized, f"{context} missing semantic marker: {marker!r}"


def _through_at_a_glance(text: str) -> str:
    """Return document start through complete ``## At a glance`` section."""
    at_glance = re.search(r"^## At a glance\s*$", text, re.MULTILINE)
    if not at_glance:
        raise AssertionError("Missing ## At a glance section")
    after_at_glance = text[at_glance.end() :]
    next_h2 = re.search(r"^## ", after_at_glance, re.MULTILINE)
    end = at_glance.end() + (next_h2.start() if next_h2 else len(after_at_glance))
    return text[:end]


def _h2_section(text: str, heading: str) -> str:
    pattern = re.compile(rf"^{re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise AssertionError(f"Missing section: {heading}")
    after = text[match.end() :]
    next_h2 = re.search(r"^## ", after, re.MULTILINE)
    end = match.end() + (next_h2.start() if next_h2 else len(after))
    return text[match.start() : end]


@pytest.fixture(scope="module")
def partners_text() -> str:
    return _read(PARTNERS_PATH)


@pytest.fixture(scope="module")
def collaboration_text() -> str:
    return _read(COLLABORATION_PATH)


@pytest.fixture(scope="module")
def faq_text() -> str:
    return _read(FAQ_PATH)


@pytest.fixture(scope="module")
def readme_text() -> str:
    return _read(README_PATH)


def test_files_and_legal_headers() -> None:
    for path in _READER_DOCS:
        assert path.is_file(), f"Missing document: {path}"
        assert _read(path).startswith(_LEGAL_HEADER), f"Missing legal header in {path.name}"


def test_issue_template_github_documentation_targets() -> None:
    texts = {}
    for path in ISSUE_TEMPLATE_PATHS:
        assert path.is_file(), f"Missing issue-template file: {path}"
        texts[path.name] = _read(path)

    for text in texts.values():
        assert _OLD_COLLABORATION_URL not in text
        assert _OLD_ROADMAP_URL not in text

    for name, text in texts.items():
        if "COLLABORATION.md" in text:
            assert _COLLABORATION_URL in text, f"{name} has a stale Collaboration URL"
    assert _ROADMAP_URL in texts["design_partner_interest.yml"]


def test_required_h1_titles(
    partners_text: str, collaboration_text: str, faq_text: str
) -> None:
    assert partners_text.splitlines()[6].strip() == _PARTNERS_H1
    assert collaboration_text.splitlines()[6].strip() == _COLLABORATION_H1
    assert faq_text.splitlines()[6].strip() == _FAQ_H1


def test_first_screen_contract(
    partners_text: str, collaboration_text: str, faq_text: str
) -> None:
    for text in _READER_DOCS:
        early = _through_at_a_glance(_read(text))
        early_norm = _normalize(early)
        for phrase in ("source-available", "active r&d", "license"):
            assert phrase in early_norm, f"Missing first-screen phrase {phrase!r}"

    for text in (partners_text, faq_text):
        early_norm = _normalize(_through_at_a_glance(text))
        assert "backend product alpha / mvp" in early_norm
        assert "partial" in early_norm


def test_partner_decision_flow_mermaid(partners_text: str) -> None:
    blocks = _MERMAID_FENCE.findall(partners_text)
    assert len(blocks) >= 1, "PARTNERS must contain at least one Mermaid block"
    forbidden_tokens = ("classDef", "style", "%%{init", "theme", "http://", "https://")
    for block in blocks:
        for token in forbidden_tokens:
            assert token not in block, f"PARTNERS: forbidden Mermaid token {token!r}"


def test_partner_decision_flow_value_first_contract(partners_text: str) -> None:
    """Current PARTNERS flow: workflow note → fit → scope/evidence → evaluation → decision."""
    block = _MERMAID_FENCE.findall(partners_text)[0]
    _assert_semantic_markers(
        block,
        (
            "workflow note",
            "fit discussion",
            "scope and evidence",
            "bounded evaluation",
            "review evidence",
            "decision",
            "continue",
            "revise",
            "stop",
        ),
        context="PARTNERS decision-flow Mermaid",
    )

    norm = _normalize(partners_text)
    for marker in (
        "fit discussion",
        "scope and evidence",
        "bounded evaluation",
        "review evidence",
        "continue / revise / stop",
    ):
        assert marker in norm, f"PARTNERS missing decision-flow marker: {marker!r}"


def test_partner_pilot_modes(partners_text: str) -> None:
    norm = _normalize(partners_text)
    for phrase in (
        "evaluation-only pilot",
        "operational or production pilot",
        "synthetic or appropriately anonymized test data",
        "explicit written permission",
    ):
        assert phrase in norm, f"PARTNERS missing pilot phrase: {phrase}"

    label_boundary = (
        "pilot" in norm
        and "sandbox" in norm
        and "proof of concept" in norm
        and "do not" in norm
        and "permission status" in norm
    )
    assert label_boundary, "PARTNERS must state pilot labels do not determine permission"


def test_pilot_brief(partners_text: str) -> None:
    norm = _normalize(partners_text)
    for phrase in (
        "concrete user workflow",
        "intended users",
        "data sources",
        "production data",
        "allowed actions",
        "forbidden actions",
        "human approvals",
        "required evidence",
        "success criteria",
        "repeated-use criteria",
        "production or commercial intent",
    ):
        assert phrase in norm, f"PARTNERS pilot brief missing: {phrase}"


def test_partner_short_first_contact_and_value_exchange(partners_text: str) -> None:
    norm = _normalize(partners_text)
    for phrase in (
        "you do not need a completed pilot brief",
        "what intergrax brings",
        "what the partner brings",
        "continue",
        "revise",
        "stop",
        "explicit written permission",
    ):
        assert phrase in norm, f"PARTNERS missing business-narrative marker: {phrase}"

    first_contact = _normalize(
        partners_text.split("## What Intergrax brings", 1)[0]
    )
    assert "3-5 sentences" in first_contact.replace("–", "-") or "workflow note" in first_contact
    assert "you do not need a completed pilot brief" in first_contact


def test_no_accidental_permission_expansion(
    partners_text: str, collaboration_text: str, faq_text: str
) -> None:
    for path, text in zip(
        ("PARTNERS", "COLLABORATION", "FAQ"),
        (partners_text, collaboration_text, faq_text),
        strict=True,
    ):
        lower = text.lower()
        for phrase in _FORBIDDEN_PERMISSION_PHRASES:
            assert phrase not in lower, f"{path} contains forbidden permission phrase: {phrase!r}"


def test_partner_exclusions(partners_text: str) -> None:
    section = _h2_section(partners_text, "## Permission and collaboration boundaries")
    section_norm = _normalize(section)
    for phrase in (
        "production rights",
        "commercial rights",
        "hosting rights",
        "redistribution rights",
        "exclusivity",
        "endorsement",
        "sla",
        "support",
        "certification",
        "compliance",
    ):
        assert phrase in section_norm, f"PARTNERS exclusions missing: {phrase}"


def test_collaboration_routing(collaboration_text: str) -> None:
    for target in (
        "EVALUATION_GUIDE.md",
        "BUILDER_QUICKSTART.md",
        "BUILD_WITH_INTERGRAX.md",
        "PARTNERS.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "SECURITY.md",
    ):
        assert target in collaboration_text, f"COLLABORATION missing link: {target}"

    norm = _normalize(collaboration_text)
    assert "a request does not grant permission" in norm
    assert "explicitly provided in writing" in norm


def test_engagement_document_ownership(
    partners_text: str, collaboration_text: str, faq_text: str
) -> None:
    evaluation = _normalize(_read(EVALUATION_GUIDE_PATH))
    builder = _normalize(_read(BUILDER_QUICKSTART_PATH))
    composition = _normalize(_read(BUILD_WITH_PATH))
    proofs = _normalize(_read(PROOFS_PATH))

    assert "self-service method" in evaluation
    assert "canonical first builder entry point" in builder
    assert "application composition plan" in composition
    assert "public proof dashboard" in proofs

    faq_norm = _normalize(faq_text)
    assert "how do i evaluate intergrax?" in faq_norm
    assert "how do i build with intergrax?" in faq_norm
    assert "evaluation guide" in faq_norm
    assert "builder quick start" in faq_norm
    assert "application composition planning" in faq_norm

    assert "evaluation guide" in _normalize(collaboration_text)
    assert "evaluation guide" in _normalize(partners_text)
    for text in (partners_text, collaboration_text, faq_text):
        assert "BUILD_WITH_INTERGRAX.md evaluation" not in text
        assert "evaluation paths" not in _normalize(text)


def test_security_boundary(collaboration_text: str) -> None:
    assert "Do not open a public issue for a suspected vulnerability." in collaboration_text
    assert "SECURITY.md" in collaboration_text


def test_faq_ownership(faq_text: str) -> None:
    for target in (
        "QUICKSTART.md",
        "LKW_PRODUCT_TOUR.md",
        "EVALUATION_GUIDE.md",
        "BUILDER_QUICKSTART.md",
        "PROOFS.md",
        "BUILD_WITH_INTERGRAX.md",
        "USE_CASES.md",
        "ROADMAP.md",
        "PARTNERS.md",
        "COLLABORATION.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "ARCHITECTURE_OVERVIEW.md",
        "DOCUMENTATION_MAP.md",
    ):
        assert target in faq_text, f"FAQ missing link: {target}"

    for heading in _FORBIDDEN_FAQ_HEADINGS:
        assert heading not in faq_text, f"FAQ contains forbidden heading: {heading}"


def test_faq_hybrid_ask_reader_route(faq_text: str) -> None:
    """PX-13: FAQ routes skeptical readers across Quick Start vs Hybrid Ask proof paths."""
    faq_norm = " ".join(_normalize(faq_text).split())
    assert "is hybrid ask proven?" in faq_norm
    for phrase in (
        "ask v1",
        "not the hybrid ask verification path",
        "bounded indexed branch through hybrid ask",
        "mixed indexed + authorized-live hybrid ask",
        "not proven",
        "production readiness",
        "complete live-provider access",
        "proofs.md",
        "lkw platform proof",
    ):
        assert phrase in faq_norm, f"FAQ Hybrid Ask route missing marker: {phrase!r}"


def test_faq_pilot_first_contact_matches_partners_lightweight_flow(faq_text: str) -> None:
    faq_norm = " ".join(_normalize(faq_text).split())
    pilot_section = _h2_section(faq_text, "## How do I discuss a pilot?")
    pilot_norm = " ".join(_normalize(pilot_section).split())
    for phrase in (
        "partners.md",
        "short",
        "workflow note",
        "sentence workflow note",
        "users",
        "workflow",
        "current problem",
        "desired result",
        "intent",
        "detailed pilot brief",
        "after mutual fit",
    ):
        assert phrase in pilot_norm, f"FAQ pilot route missing marker: {phrase!r}"
    assert "prepare a concrete pilot brief" not in pilot_norm


def test_no_internal_architecture_language(
    partners_text: str, collaboration_text: str, faq_text: str
) -> None:
    for path, text in zip(
        ("PARTNERS", "COLLABORATION", "FAQ"),
        (partners_text, collaboration_text, faq_text),
        strict=True,
    ):
        for phrase in _FORBIDDEN_ARCH_PHRASES:
            assert phrase not in text, f"{path} contains forbidden phrase: {phrase!r}"


def test_no_internal_task_ids(
    partners_text: str, collaboration_text: str, faq_text: str
) -> None:
    for doc_name, text in zip(
        ("PARTNERS", "COLLABORATION", "FAQ"),
        (partners_text, collaboration_text, faq_text),
        strict=True,
    ):
        match = _INTERNAL_TASK_PATTERN.search(text)
        assert match is None, (
            f"{doc_name} contains internal task ID: {match.group()}"
        )


def test_contact_consistency(partners_text: str, collaboration_text: str) -> None:
    assert _CONTACT_EMAIL in partners_text
    assert _CONTACT_EMAIL in collaboration_text


def test_no_planned_replacement_documents() -> None:
    for name in ("PARTNERS_AND_PILOTS.md", "LICENSE_FAQ.md"):
        assert not (REPO_ROOT / name).exists(), f"Replacement document exists: {name}"

    map_text = _read(PUBLIC_MAP_PATH)
    assert "## Planned public structure" not in map_text, (
        "PUBLIC_MAP must not contain empty planned-public-structure section"
    )
    arch_text = _read(PUBLIC_ARCHITECTURE_PATH)
    for name in ("PARTNERS_AND_PILOTS", "LICENSE_FAQ"):
        assert name not in map_text, f"PUBLIC_MAP still references planned doc: {name}"
        assert name not in arch_text, f"PUBLIC_ARCHITECTURE still references planned doc: {name}"


def test_public_map_synchronization() -> None:
    text = _read(PUBLIC_MAP_PATH)
    for doc in ("PARTNERS.md", "COLLABORATION.md", "FAQ.md", "LICENSE"):
        assert doc in text, f"PUBLIC_MAP missing reference: {doc}"
    norm = _normalize(text)
    assert "prepare a pilot" in norm
    assert "contribute or provide technical feedback" in norm
    assert "understand permission boundaries" in norm
    assert "read legally authoritative terms" in norm


def test_architecture_synchronization() -> None:
    text = _read(PUBLIC_ARCHITECTURE_PATH)
    assert "PUBLIC-DOCS-COMMERCIALIZATION-8" in text
    for phrase in (
        "Partner fit and pilot workflow",
        "Collaboration and contribution routes",
        "Practical permission-request route",
        "Legally authoritative rights and restrictions",
        "General first-contact questions",
    ):
        assert phrase in text, f"PUBLIC_ARCHITECTURE missing ownership: {phrase}"


def test_readme_compatibility(readme_text: str) -> None:
    for link in ("PARTNERS.md", "COLLABORATION.md", "LICENSE"):
        assert link in readme_text, f"README missing link: {link}"


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


def test_brevity() -> None:
    limits = {
        PARTNERS_PATH: 250,
        COLLABORATION_PATH: 180,
        FAQ_PATH: 180,
        README_PATH: 450,
    }
    for path, max_lines in limits.items():
        count = len(_read(path).splitlines())
        assert count <= max_lines, f"{path.name} has {count} lines (max {max_lines})"


def test_through_at_a_glance_eof_when_last_section() -> None:
    sample = "## Intro\n\nfoo\n\n## At a glance\n\nbar\n"
    assert _through_at_a_glance(sample) == sample


def test_through_at_a_glance_stops_at_next_h2() -> None:
    sample = "## Intro\n\n## At a glance\n\nbar\n\n## Next\n\nbaz\n"
    assert _through_at_a_glance(sample) == "## Intro\n\n## At a glance\n\nbar\n\n"


def test_at_a_glance_sections(
    partners_text: str, collaboration_text: str, faq_text: str
) -> None:
    for text in (partners_text, collaboration_text, faq_text):
        assert "## At a glance" in text
