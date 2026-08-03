# © Artur Czarnecki. All rights reserved.

"""PUBLIC-PRODUCT-EXPERIENCE-PX-1: public positioning contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
POSITIONING_PATH = REPO_ROOT / "docs" / "public-adoption" / "INTERGRAX_PUBLIC_POSITIONING.md"
ROADMAP_PATH = REPO_ROOT / "docs" / "public-adoption" / "PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md"
ARCHITECTURE_PATH = REPO_ROOT / "docs" / "public-adoption" / "PUBLIC_DOCUMENTATION_ARCHITECTURE.md"
MAINTAINER_INDEX_PATH = REPO_ROOT / "docs" / "public-adoption" / "README.md"

_LEGAL_HEADER = (
    "<!--\n"
    "© Artur Czarnecki. All rights reserved.\n"
    "Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.\n"
    "See LICENSE for permitted evaluation, collaboration, and contribution use.\n"
    "-->"
)

_CANONICAL_PRIMARY = (
    "Intergrax helps teams build AI applications that can use their knowledge and tools "
    "while keeping access, actions, and evidence under control."
)
_CANONICAL_SUPPORTING = (
    "Teams reuse shared policy, knowledge, integration, execution, and evidence foundations "
    "instead of rebuilding them for every product."
)
_CANONICAL_LKW = (
    "Local Knowledge Workspace (LKW) is the primary product path: a private-by-default "
    "workspace for adding approved knowledge sources, asking questions, and receiving "
    "grounded answers with source references and inspectable evidence."
)
_CANONICAL_STATUS = (
    "LKW is a Backend Product Alpha / MVP under active development. The current bounded "
    "product proof covers indexed knowledge workflows; complete live or hybrid access, "
    "finished end-user packaging, real-user validation, and commercial validation are not complete."
)
_CATEGORY_DESCRIPTOR = "Intergrax is a reusable foundation for governed AI applications."

_REQUIRED_SECTIONS = (
    "## At a glance",
    "## Canonical first-contact message",
    "## Product and message hierarchy",
    "## Calls to action",
    "## Audiences",
    "## Prohibited first-contact patterns",
    "## Source-of-truth boundaries",
)

_AUDIENCE_GROUPS = (
    "Potential LKW user",
    "AI engineer or developer",
    "Architect or platform engineer",
    "CTO, product lead or technical buyer",
    "Partner, integrator or design partner",
    "Contributor or deep technical reviewer",
)

_AUDIENCE_NEXT_ACTIONS = (
    "See the LKW workflow",
    "Open the bounded builder route",
    "Review architecture and proofs",
    "Assess use-case and evaluation fit",
    "Review the partner or pilot route",
    "Open the technical documentation map",
)

_FIRST_CONTACT_FORBIDDEN = (
    "Harness AI",
    "Agent OS",
    "Tier-0",
    "Tier-1",
    "Tier-2",
    "Tier-3",
    "Nexus",
    "echo.basic",
    "lab_application",
)

_TASK_ID_PATTERNS = (
    re.compile(r"\bPX-\d+\b"),
    re.compile(r"\bTOKEN-\d+\b"),
    re.compile(r"\bCTX-UCL-\d+\b"),
    re.compile(r"\bPUBLIC-DOCS-COMMERCIALIZATION-\d+\b"),
)

_FORBIDDEN_POSITIVE_CLAIMS = (
    re.compile(r"\bintergrax is production[- ]ready\.?\b", re.I),
    re.compile(r"\blkw is production[- ]ready\.?\b", re.I),
    re.compile(r"\blkw is commercially validated\.?\b", re.I),
    re.compile(r"\breal-user validation is complete\.?\b", re.I),
    re.compile(r"\breal user validation is complete\.?\b", re.I),
    re.compile(r"\bcommercial validation is complete\.?\b", re.I),
    re.compile(r"\bthe supported lkw trial is available now\.?\b", re.I),
)

_FORBIDDEN_CTA_ACTIVATION = (
    re.compile(r"\bthe current primary cta is try lkw\.?\b", re.I),
    re.compile(r"\btry lkw is available now\.?\b", re.I),
    re.compile(r"\bthe supported lkw trial is complete\.?\b", re.I),
)

_ECHO_LAB_PRIMARY_CTA_MUTATIONS = (
    "Current primary CTA: Run echo.basic.",
    "Current primary CTA: Start lab_application.",
    "The first product action is Evaluate the lab.",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_h2_section(text: str, heading_prefix: str) -> str:
    pattern = re.compile(rf"^## {re.escape(heading_prefix)}.*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise AssertionError(f"Missing ## {heading_prefix} section")
    after = text[match.end() :]
    in_fence = False
    pos = 0
    while pos < len(after):
        line_end = after.find("\n", pos)
        if line_end == -1:
            line_end = len(after)
        line = after[pos:line_end]
        if line.strip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and re.match(r"^## ", line):
            return text[match.start() : match.end() + pos]
        pos = line_end + 1 if line_end < len(after) else line_end
    return text[match.start() :]


def _claim_units(text: str) -> list[str]:
    units: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            units.extend(cell for cell in cells if cell and cell != "---")
            continue
        if re.match(r"^[-*+]\s+", stripped):
            units.append(re.sub(r"^[-*+]\s+", "", stripped))
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", stripped):
            if sentence.strip():
                units.append(sentence.strip())
    return units


def _normalize(text: str) -> str:
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _assert_no_positive_patterns(units: list[str], patterns: tuple[re.Pattern[str], ...], label: str) -> None:
    for unit in units:
        normalized = _normalize(unit)
        for pattern in patterns:
            if pattern.search(normalized):
                raise AssertionError(f"{label}: forbidden positive claim in unit: {unit!r}")


@pytest.fixture(scope="module")
def positioning_text() -> str:
    return _read(POSITIONING_PATH)


def test_positioning_file_contract(positioning_text: str) -> None:
    assert POSITIONING_PATH.is_file()
    assert positioning_text.startswith(_LEGAL_HEADER)
    assert positioning_text.splitlines()[6].strip() == "# Intergrax Public Positioning Contract"
    assert len(positioning_text.splitlines()) <= 320
    for section in _REQUIRED_SECTIONS:
        assert section in positioning_text, f"Missing section: {section}"


def test_canonical_exact_copy(positioning_text: str) -> None:
    for sentence in (
        _CANONICAL_PRIMARY,
        _CANONICAL_SUPPORTING,
        _CANONICAL_LKW,
        _CANONICAL_STATUS,
        _CATEGORY_DESCRIPTOR,
    ):
        assert sentence in positioning_text, f"Missing canonical sentence: {sentence[:50]}..."


def test_first_contact_section_forbidden_terms(positioning_text: str) -> None:
    section = _extract_h2_section(positioning_text, "Canonical first-contact message")
    for term in _FIRST_CONTACT_FORBIDDEN:
        assert term not in section, f"Forbidden term in first-contact section: {term}"
    for pattern in _TASK_ID_PATTERNS:
        assert not pattern.search(section), f"Task ID pattern in first-contact: {pattern.pattern}"


def test_route_hierarchy(positioning_text: str) -> None:
    hierarchy = _extract_h2_section(positioning_text, "Product and message hierarchy")
    norm = hierarchy.lower()
    assert "lkw = primary public product cta" in norm
    assert "token optimization = secondary platform-capability cta" in norm
    assert "product trial and platform evaluation are different routes" in norm or (
        "product trial and platform evaluation are separate" in _normalize(positioning_text)
    )
    assert "certification proof = reviewer route, not first product introduction" in norm or (
        "not first product introduction" in norm
    )


def test_cta_section(positioning_text: str) -> None:
    glance = _extract_h2_section(positioning_text, "At a glance")
    cta = _extract_h2_section(positioning_text, "Calls to action")
    assert "Current primary CTA | See the LKW workflow" in glance
    assert "Future product-trial CTA | Try LKW" in glance or "Try LKW" in glance
    assert "See the LKW workflow" in cta
    assert "Try LKW" in cta
    assert "Explore Token Optimization" in cta
    assert "Review architecture and proofs" in cta
    assert "PX-3" in cta
    assert re.search(r"try lkw may become active only after px-3", cta, re.I)
    _assert_no_positive_patterns(_claim_units(cta), _FORBIDDEN_CTA_ACTIVATION, "CTA section")
    norm = _normalize(cta)
    assert "echo.basic is not a primary public cta" in norm
    assert "labapplication is not a primary public cta" in norm or (
        "lab application is not a primary public cta" in norm
    )


def test_echo_lab_guard(positioning_text: str) -> None:
    cta = _extract_h2_section(positioning_text, "Calls to action")
    norm = cta.lower()
    assert "echo.basic" in norm and "not" in norm and "primary public cta" in norm
    assert "lab_application" in norm and "not" in norm and "primary public cta" in norm
    assert "advanced platform smoke" in norm or "maintainer diagnostics" in norm


@pytest.mark.parametrize("mutation", _ECHO_LAB_PRIMARY_CTA_MUTATIONS)
def test_echo_lab_primary_cta_mutations_raise(mutation: str) -> None:
    fake_cta = f"## Calls to action\n\n{mutation}\n"
    norm = fake_cta.lower()
    is_bad = (
        "current primary cta: run echo.basic" in norm
        or "current primary cta: start lab_application" in norm
        or "the first product action is evaluate the lab" in norm
    )
    assert is_bad


def test_audience_contract(positioning_text: str) -> None:
    section = _extract_h2_section(positioning_text, "Audiences")
    for group in _AUDIENCE_GROUPS:
        assert group in section, f"Missing audience: {group}"
    for action in _AUDIENCE_NEXT_ACTIONS:
        assert action in section, f"Missing next action: {action}"
    for pattern in _TASK_ID_PATTERNS:
        assert not pattern.search(section), f"Task ID in audience section: {pattern.pattern}"


def test_claim_safety(positioning_text: str) -> None:
    units = _claim_units(positioning_text)
    _assert_no_positive_patterns(units, _FORBIDDEN_POSITIVE_CLAIMS, "positioning")
    assert "real-user validation is incomplete" in positioning_text.lower() or (
        "real-user validation, and commercial validation are not complete" in positioning_text
    )
    assert "commercial validation is incomplete" in positioning_text.lower() or (
        "commercial validation are not complete" in positioning_text
    )


_POSITIVE_FORBIDDEN_CLAIM_MUTATIONS = (
    "Intergrax is production-ready.",
    "LKW is production-ready.",
    "LKW is commercially validated.",
    "Real-user validation is complete.",
    "Commercial validation is complete.",
    "The supported LKW trial is available now.",
)

_LEGITIMATE_INCOMPLETE_MUTATIONS = (
    "Real-user validation is incomplete.",
    "Commercial validation is incomplete.",
    "Try LKW may become active only after PX-3 provides an actual supported path.",
)


@pytest.mark.parametrize("text", _POSITIVE_FORBIDDEN_CLAIM_MUTATIONS)
def test_positive_forbidden_claim_mutations_raise(text: str) -> None:
    with pytest.raises(AssertionError):
        _assert_no_positive_patterns([text], _FORBIDDEN_POSITIVE_CLAIMS, "mutation")


@pytest.mark.parametrize("text", _LEGITIMATE_INCOMPLETE_MUTATIONS)
def test_legitimate_incomplete_mutations_pass(text: str) -> None:
    if "PX-3" in text:
        assert "PX-3" in text
        return
    _assert_no_positive_patterns([text], _FORBIDDEN_POSITIVE_CLAIMS, "mutation")


_CTA_ACTIVATION_MUTATIONS = (
    "The current primary CTA is Try LKW.",
    "Try LKW is available now.",
    "The supported LKW trial is complete.",
)


@pytest.mark.parametrize("text", _CTA_ACTIVATION_MUTATIONS)
def test_cta_activation_mutations_raise(text: str) -> None:
    with pytest.raises(AssertionError):
        _assert_no_positive_patterns([text], _FORBIDDEN_CTA_ACTIVATION, "mutation")


def test_synchronization() -> None:
    roadmap = _read(ROADMAP_PATH)
    architecture = _read(ARCHITECTURE_PATH)
    index = _read(MAINTAINER_INDEX_PATH)

    glance = _extract_h2_section(roadmap, "At a glance")
    assert "Current phase | PX-1 — READY_FOR_REVIEW" in glance
    assert "Next phase after acceptance | PX-2" in glance

    arch_norm = architecture.lower()
    assert "intergrax_public_positioning.md" in arch_norm
    assert "first-contact message" in arch_norm
    assert "lkw is the primary public product cta" in arch_norm
    assert "token optimization is the secondary capability cta" in arch_norm

    assert "PX-1 READY_FOR_REVIEW" in index or "PX-1 — READY_FOR_REVIEW" in index
    assert "Public-reader route: no" in index
    assert "INTERGRAX_PUBLIC_POSITIONING.md" in index
