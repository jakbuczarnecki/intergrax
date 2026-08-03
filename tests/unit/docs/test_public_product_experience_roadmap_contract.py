# © Artur Czarnecki. All rights reserved.

"""PUBLIC-PRODUCT-EXPERIENCE-PX-0: product experience roadmap contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
ROADMAP_PATH = REPO_ROOT / "docs" / "public-adoption" / "PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md"
ARCHITECTURE_PATH = REPO_ROOT / "docs" / "public-adoption" / "PUBLIC_DOCUMENTATION_ARCHITECTURE.md"
MAINTAINER_INDEX_PATH = REPO_ROOT / "docs" / "public-adoption" / "README.md"
PROTOCOL_PATH = REPO_ROOT / "docs" / "public-adoption" / "EXTERNAL_READER_VALIDATION_PROTOCOL.md"
ROOT_README_PATH = REPO_ROOT / "README.md"
PUBLIC_MAP_PATH = REPO_ROOT / "docs" / "PUBLIC_DOCUMENTATION_MAP.md"

_REQUIRED_ANCESTOR = "666b9ac2f78c679385ab0d34ce8bdc19a1a2fbe3"

_LEGAL_HEADER = (
    "<!--\n"
    "© Artur Czarnecki. All rights reserved.\n"
    "Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.\n"
    "See LICENSE for permitted evaluation, collaboration, and contribution use.\n"
    "-->"
)

_PHASE_HEADINGS = tuple(f"PX-{i}" for i in range(16))

_AUDIENCE_GROUPS = (
    "Potential LKW user",
    "AI engineer or developer",
    "Architect or platform engineer",
    "CTO, product lead or technical buyer",
    "Partner, integrator or design partner",
    "Contributor or deep technical reviewer",
)

_TIME_GATES = (
    "15 seconds",
    "60 seconds",
    "5 minutes",
    "15 minutes",
    "30 minutes",
    "60 minutes",
)

_FORBIDDEN_CLAIM_RULES: tuple[tuple[re.Pattern[str], tuple[re.Pattern[str], ...]], ...] = (
    (
        re.compile(r"\bexternal validation is complete\b"),
        (
            re.compile(r"\bexternal validation is not complete\b"),
            re.compile(r"\bdoes not mean external validation is complete\b"),
            re.compile(r"\bdoes not constitute external validation\b"),
            re.compile(r"\bno completed external validation\b"),
            re.compile(r"\bno current external validation is claimed\b"),
        ),
    ),
    (
        re.compile(r"\breal user validation is complete\b"),
        (
            re.compile(r"\breal user validation is not complete\b"),
            re.compile(r"\breal user validation remains incomplete\b"),
            re.compile(r"\breal-user validation is incomplete\b"),
        ),
    ),
    (
        re.compile(r"\breal-user validation is complete\b"),
        (
            re.compile(r"\breal-user validation is not complete\b"),
            re.compile(r"\breal-user validation remains incomplete\b"),
            re.compile(r"\breal-user validation is incomplete\b"),
        ),
    ),
    (
        re.compile(r"\bcommercial validation is complete\b"),
        (
            re.compile(r"\bcommercial validation is incomplete\b"),
            re.compile(r"\bcommercial validation remains incomplete\b"),
        ),
    ),
    (
        re.compile(r"\bvalidated by external users\b"),
        (
            re.compile(r"\bnot validated by external users\b"),
        ),
    ),
    (
        re.compile(r"\busers successfully completed the trial\b"),
        (
            re.compile(r"\busers have not successfully completed the trial\b"),
        ),
    ),
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
            units.extend(cell for cell in cells if cell)
            continue
        if re.match(r"^[-*+]\s+", stripped):
            units.append(re.sub(r"^[-*+]\s+", "", stripped))
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", stripped):
            if sentence.strip():
                units.append(sentence.strip())
    return units


def _normalize_claims_text(text: str) -> str:
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _span_covered(span: tuple[int, int], covered: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(c_start <= start and c_end >= end for c_start, c_end in covered)


def _assert_no_positive_forbidden_claim(text: str, path_name: str) -> None:
    for unit in _claim_units(text):
        normalized_unit = _normalize_claims_text(unit)
        for positive_re, negative_res in _FORBIDDEN_CLAIM_RULES:
            covered = [
                match.span()
                for negative_re in negative_res
                for match in negative_re.finditer(normalized_unit)
            ]
            for match in positive_re.finditer(normalized_unit):
                if not _span_covered(match.span(), covered):
                    raise AssertionError(
                        f"{path_name}: positive forbidden claim "
                        f"{match.group()!r} in unit: {unit!r}"
                    )


@pytest.fixture(scope="module")
def roadmap_text() -> str:
    return _read(ROADMAP_PATH)


def test_roadmap_file_and_header(roadmap_text: str) -> None:
    assert ROADMAP_PATH.is_file()
    assert roadmap_text.startswith(_LEGAL_HEADER)
    assert roadmap_text.splitlines()[6].strip() == "# Intergrax Public Product Experience Roadmap"
    assert len(roadmap_text.splitlines()) <= 380


def test_roadmap_baseline(roadmap_text: str) -> None:
    assert re.search(r"Baseline revision\s*\|\s*[0-9a-f]{40}", roadmap_text)
    assert "<baseline-ref>" not in roadmap_text.lower()
    assert "STARTING_REMOTE" not in roadmap_text
    if _REQUIRED_ANCESTOR in roadmap_text:
        assert "27957df0d32bdf3a7a0b07dfb92b19c891096283" in roadmap_text


def test_roadmap_at_a_glance_status(roadmap_text: str) -> None:
    glance = _extract_h2_section(roadmap_text, "At a glance")
    for row in (
        "Roadmap status | ACTIVE",
        "Current phase | PX-0 — READY_FOR_REVIEW",
        "Next phase after acceptance | PX-1",
        "External reader validation | NOT_STARTED",
        "Real-user validation | INCOMPLETE",
        "Commercial validation | INCOMPLETE",
    ):
        assert row in glance, f"At a glance missing: {row}"
    assert "PX-0 — ACCEPTED" not in roadmap_text
    assert "PX-0 — CLOSED" not in roadmap_text


def test_roadmap_phase_completeness(roadmap_text: str) -> None:
    for phase in _PHASE_HEADINGS:
        matches = list(re.finditer(rf"^## {re.escape(phase)} —", roadmap_text, re.MULTILINE))
        assert len(matches) == 1, f"Expected exactly one ## {phase} heading, found {len(matches)}"

    px0 = _extract_h2_section(roadmap_text, "PX-0 —")
    assert "READY_FOR_REVIEW" in px0
    assert "ACCEPTED" not in px0.replace("READY_FOR_REVIEW", "")

    px1 = _extract_h2_section(roadmap_text, "PX-1 —")
    assert "BLOCKED_ON_PX_0_ACCEPTANCE" in px1

    for i in range(2, 16):
        section = _extract_h2_section(roadmap_text, f"PX-{i} —")
        assert "WAITING" in section, f"PX-{i} missing WAITING status"


def test_roadmap_audience_contract(roadmap_text: str) -> None:
    section = _extract_h2_section(roadmap_text, "Audience contract")
    for group in _AUDIENCE_GROUPS:
        assert group in section, f"Missing audience: {group}"
    assert "Must understand" in section
    assert "Must be able to do" in section


def test_roadmap_experience_gates(roadmap_text: str) -> None:
    section = _extract_h2_section(roadmap_text, "First-contact success contract")
    for gate in _TIME_GATES:
        assert gate in section, f"Missing time gate: {gate}"
    norm = section.lower()
    assert "not currently claimed" in norm
    assert "15 minutes" in section


def test_roadmap_route_ownership(roadmap_text: str) -> None:
    norm = roadmap_text.lower()
    assert "lkw is the primary public product cta" in norm
    assert "token optimization is a secondary capability cta" in norm
    assert "echo.basic" in norm and "not" in norm and "primary public product demonstration" in norm
    assert "lab_application" in norm and "not" in norm and "primary public product demonstration" in norm
    assert "product trial and platform evaluation are different routes" in norm
    assert "advanced platform smoke" in norm or "maintainer diagnostics" in norm


def test_roadmap_current_vs_target(roadmap_text: str) -> None:
    norm = roadmap_text.lower()
    assert "current baseline" in norm
    assert "target experience" in norm
    assert "not currently claimed" in norm


def test_roadmap_no_fabricated_validation(roadmap_text: str) -> None:
    _assert_no_positive_forbidden_claim(roadmap_text, ROADMAP_PATH.name)

    px13 = _extract_h2_section(roadmap_text, "PX-13 —")
    px14 = _extract_h2_section(roadmap_text, "PX-14 —")
    assert "No result may be created before real sessions" in px13
    assert "No result may be created before real sessions" in px14


def test_architecture_synchronization() -> None:
    text = _read(ARCHITECTURE_PATH)
    assert "PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md" in text
    assert "Layer 5 maintainer control" in text
    assert "docs/PUBLIC_DOCUMENTATION_MAP.md" in text
    assert re.search(r"must\s+\*{0,2}not\*{0,2}\s+be\s+added", text, re.IGNORECASE)
    assert "PX-13" in text and "PX-14" in text
    assert "9B" in text and "9C" in text


def test_maintainer_index_synchronization() -> None:
    text = _read(MAINTAINER_INDEX_PATH)
    assert "PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md" in text
    assert "PX-0 READY_FOR_REVIEW" in text or "PX-0 — READY_FOR_REVIEW" in text


def test_no_public_reader_exposure() -> None:
    for path in (ROOT_README_PATH, PUBLIC_MAP_PATH):
        assert "PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md" not in _read(path)


def test_existing_validation_boundary() -> None:
    protocol = _read(PROTOCOL_PATH)
    assert "External reader validation status | NOT_STARTED" in protocol or (
        "External reader validation" in protocol and "NOT_STARTED" in protocol
    )


_POSITIVE_FORBIDDEN_CLAIM_MUTATIONS = (
    "External validation is complete.",
    "Real-user validation is complete.",
    "Commercial validation is complete.",
    "Validated by external users.",
    "Users successfully completed the trial.",
)

_LEGITIMATE_NEGATIVE_CLAIM_MUTATIONS = (
    "External validation is not complete.",
    "Real-user validation is incomplete.",
    "Commercial validation is incomplete.",
    "No current external validation is claimed.",
)


@pytest.mark.parametrize("text", _POSITIVE_FORBIDDEN_CLAIM_MUTATIONS)
def test_positive_forbidden_claim_mutations_raise(text: str) -> None:
    with pytest.raises(AssertionError):
        _assert_no_positive_forbidden_claim(text, "mutation")


@pytest.mark.parametrize("text", _LEGITIMATE_NEGATIVE_CLAIM_MUTATIONS)
def test_legitimate_negative_claim_mutations_pass(text: str) -> None:
    _assert_no_positive_forbidden_claim(text, "mutation")
