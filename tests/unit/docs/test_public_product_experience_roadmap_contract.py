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

_REQUIRED_ANCESTOR = "9c423d0ff760fa6e574a3977e5c6fd2af2a5a95d"

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
        "Current phase | PX-1 — READY_FOR_REVIEW",
        "Next phase after acceptance | PX-2",
        "External reader validation | NOT_STARTED",
        "Real-user validation | INCOMPLETE",
        "Commercial validation | INCOMPLETE",
    ):
        assert row in glance, f"At a glance missing: {row}"
    assert "PX-1 — ACCEPTED" not in roadmap_text
    assert "PX-1 — CLOSED" not in roadmap_text


def test_roadmap_phase_completeness(roadmap_text: str) -> None:
    for phase in _PHASE_HEADINGS:
        matches = list(re.finditer(rf"^## {re.escape(phase)} —", roadmap_text, re.MULTILINE))
        assert len(matches) == 1, f"Expected exactly one ## {phase} heading, found {len(matches)}"

    px0 = _extract_h2_section(roadmap_text, "PX-0 —")
    assert "ACCEPTED / CLOSED" in px0
    assert "9c423d0ff760fa6e574a3977e5c6fd2af2a5a95d" in px0

    px1 = _extract_h2_section(roadmap_text, "PX-1 —")
    assert "READY_FOR_REVIEW" in px1
    assert "ACCEPTED" not in px1.replace("READY_FOR_REVIEW", "")

    px2 = _extract_h2_section(roadmap_text, "PX-2 —")
    assert "BLOCKED_ON_PX_1_ACCEPTANCE" in px2

    for i in range(3, 16):
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


_EXTERNAL_VALIDATION_SECTION_HEADING = "External-validation boundary"

_EXTERNAL_VALIDATION_REQUIRED_TOKENS = (
    "PX-13",
    "PX-14",
    "PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md",
    "paused",
    "superseded",
    "historical",
)

_STALE_ACTIVE_9B_9C_PATTERNS = (
    re.compile(r"not complete until real external-reader sessions\s*\(\s*9B\s*\)", re.I),
    re.compile(r"not complete until.*9B.*9C", re.I),
    re.compile(r"Point 9.*not complete until.*9B", re.I),
)

_ACTIVE_LEGACY_9B_9C_PATTERNS = (
    re.compile(r"9B and 9C remain the active", re.I),
    re.compile(r"9B and 9C are the active", re.I),
    re.compile(r"complete 9B and 9C", re.I),
    re.compile(r"then complete 9B", re.I),
    re.compile(r"active external-validation phases", re.I),
)

_ALLOWED_9B_9C_SENTENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"the previous planned 9b and 9c execution steps are "
        r"(?:paused and superseded|superseded and paused|paused and replaced|"
        r"replaced and paused|replaced|superseded) "
        r"by px-13 and px-14 of public_product_experience_roadmap\.md\.?"
    ),
    re.compile(
        r"the historical names 9b and 9c do not define an additional "
        r"active execution path\.?"
    ),
)


def _normalize_boundary_sentence(sentence: str) -> str:
    sentence = re.sub(r"[*`]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip().lower()
    return sentence


def _boundary_section_sentences(section: str) -> list[str]:
    prose_lines = [
        line.strip()
        for line in section.splitlines()
        if line.strip() and not line.strip().startswith("##")
    ]
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(prose_lines))
        if sentence.strip()
    ]


def _valid_external_validation_boundary_section(
    *,
    additional_sentence: str | None = None,
) -> str:
    lines = [
        f"## {_EXTERNAL_VALIDATION_SECTION_HEADING}",
        "",
        (
            "The previous planned 9B and 9C execution steps are paused and "
            "superseded by PX-13 and PX-14 of "
            "PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md."
        ),
    ]
    if additional_sentence is not None:
        lines.extend(["", additional_sentence])
    lines.extend(
        [
            "",
            "PX-13 owns real external comprehension and trial sessions.",
            "",
            "PX-14 owns findings, corrections and required reruns.",
            "",
            (
                "The historical names 9B and 9C do not define an additional "
                "active execution path."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _validate_external_validation_boundary_section(section: str) -> None:
    lowered = section.lower()
    for token in _EXTERNAL_VALIDATION_REQUIRED_TOKENS:
        assert token.lower() in lowered, f"Missing required token: {token}"
    assert re.search(
        r"not define an additional active execution path",
        section,
        re.IGNORECASE,
    ), "Missing additional-active-path negation"
    assert re.search(r"PX-13.*owns", section, re.IGNORECASE | re.DOTALL), (
        "PX-13 must own real external sessions"
    )
    assert re.search(r"PX-14.*owns", section, re.IGNORECASE | re.DOTALL), (
        "PX-14 must own findings, corrections and reruns"
    )
    for pattern in _STALE_ACTIVE_9B_9C_PATTERNS:
        assert not pattern.search(section), f"Stale active 9B/9C rule found: {pattern.pattern}"
    for pattern in _ACTIVE_LEGACY_9B_9C_PATTERNS:
        assert not pattern.search(section), (
            f"Active legacy 9B/9C wording found: {pattern.pattern}"
        )
    for sentence in _boundary_section_sentences(section):
        if not re.search(r"\b9[BC]\b", sentence, re.IGNORECASE):
            continue
        normalized = _normalize_boundary_sentence(sentence)
        assert any(
            pattern.fullmatch(normalized) for pattern in _ALLOWED_9B_9C_SENTENCE_PATTERNS
        ), f"9B/9C sentence not on historical allowlist: {sentence!r}"


def test_architecture_synchronization() -> None:
    text = _read(ARCHITECTURE_PATH)
    assert "PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md" in text
    assert "INTERGRAX_PUBLIC_POSITIONING.md" in text
    assert "Layer 5 maintainer control" in text
    assert "docs/PUBLIC_DOCUMENTATION_MAP.md" in text
    assert re.search(r"must\s+\*{0,2}not\*{0,2}\s+be\s+added", text, re.IGNORECASE)
    norm = text.lower()
    assert "lkw is the primary public product cta" in norm
    assert "token optimization is the secondary capability cta" in norm
    section = _extract_h2_section(text, _EXTERNAL_VALIDATION_SECTION_HEADING)
    _validate_external_validation_boundary_section(section)


def test_maintainer_index_synchronization() -> None:
    text = _read(MAINTAINER_INDEX_PATH)
    assert "PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md" in text
    assert "PX-1 READY_FOR_REVIEW" in text or "PX-1 — READY_FOR_REVIEW" in text
    assert "INTERGRAX_PUBLIC_POSITIONING.md" in text
    assert "Public-reader route: no" in text


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


_NEGATIVE_EXTERNAL_VALIDATION_BOUNDARY_MUTATIONS = (
    "Point 9 is not complete until 9B and 9C are done.",
    "Run PX-13 and PX-14, then complete 9B and 9C.",
    "9B and 9C remain the active external-validation phases.",
    "Historical 9B and 9C should still be completed after PX-14.",
    "The paused 9B and 9C will run after PX-14.",
    "Historical 9B and 9C remain required before external validation is complete.",
    "9B and 9C were superseded, but they must still be executed after PX-14.",
    "After PX-14, complete the historical 9B and 9C steps.",
    (
        "PX-13 and PX-14 are active, while historical 9B and 9C remain "
        "future work."
    ),
)

_POSITIVE_EXTERNAL_VALIDATION_BOUNDARY_MUTATIONS = (
    (
        "The previous planned 9B and 9C execution steps are paused and "
        "superseded by PX-13 and PX-14 of PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md."
    ),
    (
        "The historical names 9B and 9C do not define an additional "
        "active execution path."
    ),
)


def test_valid_external_validation_boundary_section_passes() -> None:
    _validate_external_validation_boundary_section(
        _valid_external_validation_boundary_section()
    )


@pytest.mark.parametrize("text", _NEGATIVE_EXTERNAL_VALIDATION_BOUNDARY_MUTATIONS)
def test_external_validation_boundary_negative_mutations_raise(text: str) -> None:
    section = _valid_external_validation_boundary_section(additional_sentence=text)
    with pytest.raises(AssertionError, match="9B/9C|historical|legacy"):
        _validate_external_validation_boundary_section(section)


@pytest.mark.parametrize("text", _POSITIVE_EXTERNAL_VALIDATION_BOUNDARY_MUTATIONS)
def test_external_validation_boundary_positive_mutations_pass(text: str) -> None:
    section = _valid_external_validation_boundary_section(additional_sentence=text)
    _validate_external_validation_boundary_section(section)
