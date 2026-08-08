# © Artur Czarnecki. All rights reserved.

"""PUBLIC-DOCS-COMMERCIALIZATION-9A: external reader validation contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_GUIDE_PATH = REPO_ROOT / "docs" / "project" / "builders" / "EVALUATION_GUIDE.md"
PROTOCOL_PATH = REPO_ROOT / "docs" / "project" / "maintainers" / "public-adoption" / "EXTERNAL_READER_VALIDATION_PROTOCOL.md"
LAUNCH_CHECKLIST_PATH = REPO_ROOT / "docs" / "project" / "maintainers" / "public-adoption" / "PUBLIC_LAUNCH_CHECKLIST.md"
PX_ROADMAP_PATH = REPO_ROOT / "docs" / "project" / "overview" / "PUBLIC_PRODUCT_EXPERIENCE_ROADMAP.md"
OUTREACH_KIT_PATH = REPO_ROOT / "docs" / "project" / "maintainers" / "public-adoption" / "OUTREACH_KIT.md"
PUBLIC_ADOPTION_INDEX_PATH = REPO_ROOT / "docs" / "project" / "maintainers" / "public-adoption" / "README.md"
PUBLIC_ARCHITECTURE_PATH = REPO_ROOT / "docs" / "project" / "maintainers" / "public-adoption" / "PUBLIC_DOCUMENTATION_ARCHITECTURE.md"
README_PATH = REPO_ROOT / "README.md"
FAQ_PATH = REPO_ROOT / "docs" / "project" / "overview" / "FAQ.md"
WHY_PATH = REPO_ROOT / "docs" / "project" / "overview" / "WHY_INTERGRAX.md"
USE_CASES_PATH = REPO_ROOT / "docs" / "project" / "overview" / "USE_CASES.md"
ROADMAP_PATH = REPO_ROOT / "docs" / "project" / "overview" / "ROADMAP.md"
BUILD_PATH = REPO_ROOT / "docs" / "project" / "builders" / "BUILD_WITH_INTERGRAX.md"
PROOFS_PATH = REPO_ROOT / "docs" / "project" / "proofs" / "PROOFS.md"
PARTNERS_PATH = REPO_ROOT / "docs" / "project" / "community" / "PARTNERS.md"
COLLABORATION_PATH = REPO_ROOT / "docs" / "project" / "community" / "COLLABORATION.md"
LICENSE_PATH = REPO_ROOT / "LICENSE"
LKW_PROOF_PATH = REPO_ROOT / "docs" / "project" / "proofs" / "LKW_PLATFORM_PROOF.md"
TOKEN_GUIDE_PATH = REPO_ROOT / "docs" / "project" / "capabilities" / "token_optimization" / "README.md"

_LEGAL_HEADER = (
    "<!--\n"
    "© Artur Czarnecki. All rights reserved.\n"
    "Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.\n"
    "See LICENSE for permitted evaluation, collaboration, and contribution use.\n"
    "-->"
)

_PRINCIPAL_DOCS = (
    EVALUATION_GUIDE_PATH,
    PROTOCOL_PATH,
    LAUNCH_CHECKLIST_PATH,
    OUTREACH_KIT_PATH,
)

_H1_BY_PATH = {
    EVALUATION_GUIDE_PATH: "# Intergrax Evaluation Guide",
    PROTOCOL_PATH: "# External Reader Validation Protocol",
    LAUNCH_CHECKLIST_PATH: "# Intergrax Public Launch Checklist",
    OUTREACH_KIT_PATH: "# Intergrax Outreach Kit",
}

def _does_not_mean_list_pattern(claim: str) -> re.Pattern[str]:
    return re.compile(rf"\bdoes not mean:[^.!?]*\b{claim}\b")


_FORBIDDEN_CLAIM_RULES: tuple[tuple[re.Pattern[str], tuple[re.Pattern[str], ...]], ...] = (
    (
        re.compile(r"\bexternal validation is complete\b"),
        (
            re.compile(r"\bexternal validation is not complete\b"),
            re.compile(r"\bdoes not mean external validation is complete\b"),
            re.compile(r"\bdoes not constitute external validation\b"),
            re.compile(r"\bno completed external validation\b"),
            _does_not_mean_list_pattern(r"external validation is complete"),
        ),
    ),
    (
        re.compile(r"\bvalidated by external users\b"),
        (
            re.compile(r"\bnot validated by external users\b"),
            _does_not_mean_list_pattern(r"validated by external users"),
        ),
    ),
    (
        re.compile(r"\breal user validation is complete\b"),
        (
            re.compile(r"\breal user validation is not complete\b"),
            re.compile(r"\breal user validation remains incomplete\b"),
            _does_not_mean_list_pattern(r"real user validation is complete"),
        ),
    ),
    (
        re.compile(r"\breal user validation complete\b"),
        (
            re.compile(r"\breal user validation is not complete\b"),
            re.compile(r"\breal user validation remains incomplete\b"),
            _does_not_mean_list_pattern(r"real user validation complete"),
        ),
    ),
    (
        re.compile(r"\bcommercial validation is complete\b"),
        (
            re.compile(r"\bcommercial validation is incomplete\b"),
            re.compile(r"\bcommercial validation remains incomplete\b"),
            _does_not_mean_list_pattern(r"commercial validation is complete"),
        ),
    ),
    (
        re.compile(r"\bcommercially validated\b"),
        (
            re.compile(r"\bnot commercially validated\b"),
            _does_not_mean_list_pattern(r"commercially validated"),
        ),
    ),
    (
        re.compile(r"\bintergrax is commercially validated\b"),
        (
            re.compile(r"\bintergrax is not commercially validated\b"),
            _does_not_mean_list_pattern(r"intergrax is commercially validated"),
        ),
    ),
    (
        re.compile(r"\bintergrax is production ready\b"),
        (
            re.compile(r"\bintergrax is not production ready\b"),
            _does_not_mean_list_pattern(r"intergrax is production ready"),
        ),
    ),
    (
        re.compile(r"\bproduction ready\b"),
        (
            re.compile(r"\bnot production ready\b"),
            re.compile(r"\bwithout claiming production readiness\b"),
            _does_not_mean_list_pattern(r"production ready"),
        ),
    ),
    (
        re.compile(r"\bproduct validated\b"),
        (
            re.compile(r"\bnot product validated\b"),
            _does_not_mean_list_pattern(r"product validated"),
        ),
    ),
    (
        re.compile(r"\busability certified\b"),
        (
            re.compile(r"\bnot usability certified\b"),
            _does_not_mean_list_pattern(r"usability certified"),
        ),
    ),
)

_STALE_PHRASES = (
    "Agent OS",
    "Nexus",
    "Tier-0",
    "Tier-1",
    "Tier-2",
    "Tier-3",
    "README.md#start-here",
    "README.md#proof-of-platform",
    "LOCAL_KNOWLEDGE_WORKSPACE_ALPHA",
    "docs/project/proofs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md",
)

_ISSUE_URL_PATTERN = re.compile(
    r"github\.com/jakbuczarnecki/intergrax/issues/\d+",
    re.IGNORECASE,
)

_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

_INTERNAL_TASK_PATTERN = re.compile(
    r"(CTX-UCL-|TOKEN-10|LKW-SLACK-|GOOGLE-WORKSPACE-|"
    r"MSGRAPH-|PUBLIC-DOCS-COMMERCIALIZATION-)",
    re.IGNORECASE,
)
_LKW_IMPLEMENTATION_TASK_PATTERN = re.compile(
    r"\bLKW-[A-Z0-9]+(?:-[A-Z0-9]+)+-\d+\b",
    re.IGNORECASE,
)

_MOVING_REPO_URL = re.compile(
    r"https?://github\.com/jakbuczarnecki/intergrax/?(?:\s|$|\))",
    re.IGNORECASE,
)

_RELATIVE_PARTICIPANT_LINK = re.compile(r"\]\(\.\./")

_SESSION_TEMPLATE_METADATA_FIELDS = (
    "Session ID",
    "Validation wave",
    "Date",
    "Pinned commit or immutable tag",
    "Participant-facing root URL",
    "Primary cohort",
    "Prior familiarity",
    "Tracks attempted",
    "Track B environment",
    "Consent for quotation",
)

_SESSION_TEMPLATE_EVIDENCE_FIELDS = (
    "Participant one-sentence description",
    "Identified strongest product proof",
    "Stated product-proof maturity",
    "Wrong or uncertain conclusions",
    "First navigation route",
    "Dead ends",
    "Moderator interventions",
    "Broken links",
    "Technical errors",
    "Follow-up notes",
)

_INVITATION_SECTIONS = (
    (
        "Blind first-contact invitation",
        ("<pinned-repository-root-url>", "<pinned-ref>"),
    ),
    (
        "Technical evaluation invitation",
        ("<pinned-evaluation-guide-url>", "<pinned-ref>"),
    ),
    (
        "LKW workflow-fit invitation",
        (
            "<pinned-lkw-proof-url>",
            "<pinned-use-cases-url>",
            "<pinned-partners-url>",
            "<pinned-ref>",
        ),
    ),
    (
        "Architecture and governance review invitation",
        (
            "<pinned-architecture-url>",
            "<pinned-proofs-url>",
            "<pinned-build-url>",
            "<pinned-ref>",
        ),
    ),
)

_EVAL_GUIDE_LINKS = (
    "README.md",
    "../overview/FAQ.md",
    "../overview/WHY_INTERGRAX.md",
    "../overview/USE_CASES.md",
    "../overview/ROADMAP.md",
    "BUILD_WITH_INTERGRAX.md",
    "../proofs/PROOFS.md",
    "../architecture/ARCHITECTURE_OVERVIEW.md",
    "../community/PARTNERS.md",
    "../community/COLLABORATION.md",
    "../../../LICENSE",
    "../community/PUBLIC_DOCUMENTATION_MAP.md",
    "../technical/DOCUMENTATION_MAP.md",
    "../proofs/LKW_PLATFORM_PROOF.md",
    "../capabilities/token_optimization/README.md",
)

_TRACK_A_TASKS = (
    "Explain Intergrax in one sentence",
    "Identify the strongest current product proof",
    "State the maturity of that product proof",
    "Find where to decide whether a use case fits",
    "Find where to begin technical evaluation",
    "Determine whether production or commercial use is automatically permitted",
    "Find how to discuss a pilot or partnership",
    "State the next action you would take",
)

_EVIDENCE_FIELDS = (
    "anonymized session ID",
    "pinned commit or tag",
    "participant cohort",
    "prior familiarity",
    "validation tracks",
    "task result for every mandatory task",
    "first navigation route",
    "dead ends",
    "moderator interventions",
    "broken links",
    "finding severity",
    "consent status",
)

_PRIVACY_PROHIBITIONS = (
    "names",
    "email addresses",
    "raw recordings",
    "employer-confidential data",
)

_FABRICATED_PATTERNS = (
    re.compile(r"\|\s*S-\d{3}\s*\|", re.IGNORECASE),
    re.compile(r"participant:\s*[A-Z][a-z]+\s+[A-Z][a-z]+", re.IGNORECASE),
    re.compile(r"total sessions\s*\|\s*[1-9]\d*\s*\|", re.IGNORECASE),
)

_LINK_CHECK_PATHS = (
    EVALUATION_GUIDE_PATH,
    PROTOCOL_PATH,
    LAUNCH_CHECKLIST_PATH,
    OUTREACH_KIT_PATH,
    PUBLIC_ADOPTION_INDEX_PATH,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalize(text: str) -> str:
    return re.sub(r"[*_`]", "", text).lower()


def _normalize_claims_text(text: str) -> str:
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()


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


def _extract_h2_section(text: str, heading: str) -> str:
    pattern = re.compile(rf"^## {re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise AssertionError(f"Missing ## {heading} section")
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


def _through_at_a_glance(text: str) -> str:
    at_glance = re.search(r"^## At a glance\s*$", text, re.MULTILINE)
    if not at_glance:
        raise AssertionError("Missing ## At a glance section")
    after = text[at_glance.end() :]
    next_h2 = re.search(r"^## ", after, re.MULTILINE)
    if not next_h2:
        raise AssertionError("Missing H2 section after ## At a glance")
    return text[: at_glance.end() + next_h2.start()]


@pytest.fixture(scope="module")
def protocol_text() -> str:
    return _read(PROTOCOL_PATH)


@pytest.fixture(scope="module")
def evaluation_text() -> str:
    return _read(EVALUATION_GUIDE_PATH)


@pytest.fixture(scope="module")
def outreach_text() -> str:
    return _read(OUTREACH_KIT_PATH)


@pytest.fixture(scope="module")
def launch_text() -> str:
    return _read(LAUNCH_CHECKLIST_PATH)


def test_files_and_legal_headers() -> None:
    for path in _PRINCIPAL_DOCS:
        assert path.is_file(), f"Missing document: {path}"
        assert _read(path).startswith(_LEGAL_HEADER), f"Missing legal header in {path.name}"


def test_h1_titles() -> None:
    for path, expected_h1 in _H1_BY_PATH.items():
        assert _read(path).splitlines()[6].strip() == expected_h1, f"Wrong H1 in {path.name}"


def test_protocol_first_screen_status(protocol_text: str) -> None:
    opening = _through_at_a_glance(protocol_text)
    for phrase in ("READY_TO_RUN", "NOT_STARTED", "Minimum completed sessions", "Raw personal data"):
        assert phrase in opening, f"Protocol At a glance missing: {phrase}"
    assert "5" in opening
    assert "Creating this protocol does not mean external validation is complete." in protocol_text


def test_no_fictional_completion_claim(protocol_text: str, evaluation_text: str) -> None:
    for path, text in ((PROTOCOL_PATH.name, protocol_text), (EVALUATION_GUIDE_PATH.name, evaluation_text)):
        _assert_no_positive_forbidden_claim(text, path)


def test_reviewer_cohorts(protocol_text: str) -> None:
    section = protocol_text
    assert "minimum of **five completed independent sessions**" in section or "minimum of five completed independent sessions" in _normalize(section)
    for phrase in (
        "at least 2",
        "Technical readers unfamiliar",
        "Potential LKW or governed-knowledge",
        "Architecture, platform, governance or observability",
        "one primary cohort only",
        "do not count",
        "qualitative and not statistically representative",
    ):
        assert phrase in section or phrase.lower() in _normalize(section), f"Missing cohort rule: {phrase}"


def test_validation_tracks(protocol_text: str) -> None:
    for track in ("Track A", "Track B", "Track C"):
        assert track in protocol_text
    assert "15 minutes" in protocol_text
    assert "30–60 minutes" in protocol_text
    assert "Mandatory for every participant" in protocol_text
    assert "at least two completed sessions" in protocol_text.lower()


def test_mandatory_tasks(protocol_text: str) -> None:
    for task in _TRACK_A_TASKS:
        assert task in protocol_text, f"Missing mandatory task: {task}"


def test_facilitation_protections(protocol_text: str) -> None:
    norm = _normalize(protocol_text)
    for phrase in (
        "pinned repository revision",
        "same mandatory prompts",
        "no coaching",
        "no expected answers",
        "record",
        "intervention",
        "no correction of wrong conclusions until the end",
        "same pinned commit or immutable tag",
        "moving default-branch url",
        "selected before recruitment",
        "creates a new validation wave",
        "must match the content actually shown",
    ):
        assert phrase in norm, f"Missing facilitation rule: {phrase}"


def test_scoring_and_severity(protocol_text: str) -> None:
    for score in ("PASS", "FRICTION", "FAIL", "NOT_RUN"):
        assert score in protocol_text
    for severity in ("CRITICAL", "MAJOR", "MINOR", "OBSERVATION"):
        assert severity in protocol_text


def test_evidence_and_privacy(protocol_text: str) -> None:
    norm = _normalize(protocol_text)
    for field in _EVIDENCE_FIELDS:
        assert field.lower() in norm, f"Missing evidence field: {field}"
    for prohibition in _PRIVACY_PROHIBITIONS:
        assert prohibition in norm, f"Missing privacy prohibition: {prohibition}"
    assert "anonymized" in norm
    assert "consent" in norm


def test_completion_gates(protocol_text: str) -> None:
    norm = _normalize(protocol_text)
    for gate in (
        "at least five completed sessions",
        "all required cohorts are represented",
        "every participant attempted all track a tasks",
        "at least 80%",
        "no unresolved",
        "major",
        "at least two track b sessions",
        "pinned revision is recorded",
        "anonymized aggregate summary",
        "rerun after corrections",
    ):
        assert gate in norm, f"Missing completion gate: {gate}"
    assert "VALIDATED_FOR_BOUNDED_OUTREACH" in protocol_text
    assert "does not mean" in norm


def test_empty_templates_only(protocol_text: str) -> None:
    assert "Session record template" in protocol_text
    assert "Aggregate summary template" in protocol_text
    for placeholder in ("<session-id>", "<date>", "<pinned-ref>"):
        assert placeholder in protocol_text
    for pattern in _FABRICATED_PATTERNS:
        assert not pattern.search(protocol_text), f"Fabricated data in protocol: {pattern.pattern}"


def test_evaluation_guide_positioning(evaluation_text: str) -> None:
    opening = evaluation_text[: evaluation_text.find("## What does a bounded evaluation mean?")]
    norm = _normalize(opening)
    for phrase in (
        "source-available",
        "active r&d",
        "primary product proof",
        "backend product alpha",
        "partial",
        "real-user validation",
        "commercial validation",
        "license",
    ):
        assert phrase in norm, f"EVALUATION_GUIDE missing positioning: {phrase}"
    assert "## At a glance" in evaluation_text


def test_evaluation_guide_bounded_evaluation_contract(evaluation_text: str) -> None:
    norm = _normalize(evaluation_text)
    for phrase in (
        "bounded evaluation is a reproducible attempt",
        "one stated claim or workflow",
        "choose one evaluation target",
        "pinned repository revision",
        "claim being tested",
        "what the path does not prove",
        "canonical owner",
        "inspect evidence",
        "evaluation target:",
        "expected result:",
        "observed result:",
        "skipped/unavailable:",
        "known limitation:",
        "proceed",
        "defer",
        "stop",
        "self-service evaluation does not constitute external reader validation",
        "builder quick start",
        "buildwithintergrax",
        "proofs",
        "quick start pass alone is not full technical validation",
        "no validated evaluation duration is promised",
    ):
        assert phrase in norm, f"EVALUATION_GUIDE missing contract invariant: {phrase}"

    for target in (
        "lkw product trial",
        "lkw deep product/platform proof",
        "token optimization capability",
        "architecture / builder fit",
    ):
        assert target in norm, f"EVALUATION_GUIDE missing target: {target}"

    assert "repository baseline / broader confidence check" in norm
    assert "not proof of the selected product or capability claim" in norm
    assert "external reader validation protocol" in norm
    assert "what is currently demonstrated" in norm
    assert "how to test one claim fairly" in norm
    for stale_heading in (
        "## 5-minute orientation",
        "## 15-minute problem, fit and direction",
        "## 30-minute bounded technical evaluation",
        "## 45–60 minute deep evaluation",
    ):
        assert stale_heading not in evaluation_text
    assert "time-boxed" not in norm


def test_evaluation_routes(evaluation_text: str) -> None:
    for link in _EVAL_GUIDE_LINKS:
        assert link in evaluation_text, f"EVALUATION_GUIDE missing link: {link}"


def test_no_stale_reader_language(evaluation_text: str, outreach_text: str) -> None:
    for path, text in (
        (EVALUATION_GUIDE_PATH.name, evaluation_text),
        (OUTREACH_KIT_PATH.name, outreach_text),
    ):
        for phrase in _STALE_PHRASES:
            assert phrase not in text, f"{path} contains stale phrase: {phrase!r}"
        assert not _ISSUE_URL_PATTERN.search(text), f"{path} contains hard-coded issue URL"


def test_launch_checklist_readiness(launch_text: str) -> None:
    for section in (
        "Audit baseline",
        "Reader journey readiness",
        "PX-12 internal readiness evidence",
        "Claims and legal boundary review",
        "PX-13 validation-wave preparation",
    ):
        assert section in launch_text, f"Launch checklist missing section: {section}"
    assert "NOT_STARTED" in launch_text
    norm = _normalize(launch_text)
    assert "accepted / closed" in norm
    assert "blocked_on_px_12_acceptance" not in norm
    assert "Status:\nNOT_STARTED" in launch_text
    assert "External reader validation:\nNOT_STARTED" in launch_text
    for state in (
        "PX-12:\nACCEPTED / CLOSED",
        "PRE-PX13 completion gate:\nIN_PROGRESS",
        "PX-13:\nNOT_STARTED / BLOCKED ON PRE-PX13 COMPLETION",
    ):
        assert state in launch_text, f"Launch checklist missing readiness state: {state}"
    assert "Wave 1 preparation is not authorized" in launch_text
    assert "does not mean external validation is complete" in norm
    assert "checklist completion does not conduct sessions" in norm or (
        "no fictional session" in norm
    )


def test_pre_px13_gate_blocks_external_validation() -> None:
    roadmap_text = _read(PX_ROADMAP_PATH)
    roadmap_norm = _normalize(roadmap_text)
    gate = _extract_h2_section(
        roadmap_text,
        "PRE-PX13 — Product, Proof & Public Experience Completion Gate",
    )
    gate_norm = _normalize(gate)

    assert "Current program gate | PRE-PX13 — IN_PROGRESS" in roadmap_text
    assert "Next external phase | PX-13 — BLOCKED_ON_PRE_PX13_COMPLETION" in roadmap_text
    assert "External reader validation | NOT_STARTED" in roadmap_text
    assert "## PX-12" in roadmap_text and "Status:** ACCEPTED / CLOSED" in roadmap_text
    assert "**Status:** NOT_STARTED / BLOCKED_ON_PRE_PX13_COMPLETION" in roadmap_text
    assert "external reader validation remains notstarted" in gate_norm
    for condition in (
        "complete the selected lkw product experience",
        "converge claims and evidence",
        "verify runnable claims",
        "stop-and-fix friction rule",
        "verify restart and recovery claims",
        "complete deployment and onboarding",
        "pass final product acceptance proof",
        "synchronize public claims",
        "complete the visual experience review",
        "complete a clean-room internal user walkthrough",
        "no known intentional pre-external rewrite",
    ):
        assert condition in gate_norm, f"PRE-PX13 gate missing condition: {condition}"
    assert "no fictional" in roadmap_norm
    assert not _LKW_IMPLEMENTATION_TASK_PATTERN.search(gate), (
        "PRE-PX13 gate mirrors detailed LKW implementation task IDs"
    )
    assert not re.search(r"\|\s*S-\d{3}\s*\|", gate, re.IGNORECASE)


def test_outreach_templates(outreach_text: str) -> None:
    for section in (
        "Blind first-contact invitation",
        "Technical evaluation invitation",
        "LKW workflow-fit invitation",
        "Architecture and governance review invitation",
        "Moderator opening",
        "Post-session questions",
        "Privacy and quotation",
        "What not to say",
    ):
        assert section in outreach_text, f"Outreach kit missing section: {section}"
    norm = _normalize(outreach_text)
    assert "endorsement" in norm
    assert "not a request for endorsement" in norm or "no endorsement" in norm or "not endorsement" in norm


def test_maintainer_index_synchronization() -> None:
    text = _read(PUBLIC_ADOPTION_INDEX_PATH)
    assert "External Reader Validation Protocol" in text
    assert "Protocol status" in text
    assert "READY_TO_RUN" in text
    assert "External reader validation" in text
    assert "NOT_STARTED" in text
    assert "PUBLIC_DOCUMENTATION_MAP.md" in text
    assert "not the default first-contact path" in text.lower() or "not a normal external first-contact route" in text.lower()


def test_architecture_synchronization() -> None:
    text = _read(PUBLIC_ARCHITECTURE_PATH)
    assert "PUBLIC-DOCS-COMMERCIALIZATION-9A" in text
    for phrase in (
        "External reader validation methodology",
        "Pre-session and pre-outreach readiness",
        "Participant recruitment and session-request templates",
    ):
        assert phrase in text, f"Architecture missing: {phrase}"
    assert "does not constitute external validation" in text.lower()


def test_documentation_ownership_model() -> None:
    normalized = " ".join(_normalize(_read(PUBLIC_ARCHITECTURE_PATH)).split())
    for layer in (
        "project documentation",
        "project projections",
        "module sources of truth",
    ):
        assert layer in normalized, f"Missing ownership layer: {layer}"

    for invariant in (
        "accepted module evidence",
        "unit tests alone is insufficient",
        "does not globally block unrelated project documentation work",
        "implementation roadmaps remain module-owned and are not public claim dashboards",
    ):
        assert invariant in normalized, f"Missing ownership invariant: {invariant}"

    evidence_packet_fields = (
        "capability",
        "status",
        "accepted sha",
        "proven",
        "user-visible outcome",
        "not proven",
        "verification path",
        "public claim candidate",
        "visual opportunity",
        "public docs potentially affected",
    )
    for field in evidence_packet_fields:
        assert field in normalized, f"Missing evidence handoff field: {field}"

    pipeline = (
        "module implementation",
        "module acceptance",
        "accepted evidence",
        "project projection",
        "optional readme promotion",
    )
    ownership_section = normalized[normalized.index("## non-blocking rule and claim promotion") :]
    positions = [ownership_section.index(step) for step in pipeline]
    assert positions == sorted(positions), "Claim promotion pipeline is out of order"


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
            if clean.startswith("<") and clean.endswith(">"):
                continue
            resolved = (base / clean).resolve()
            assert resolved.exists(), f"Broken link in {doc_path.name}: {target}"


def test_session_record_template_contract(protocol_text: str) -> None:
    section = _extract_h2_section(protocol_text, "Session record template")
    for field in _SESSION_TEMPLATE_METADATA_FIELDS:
        assert field in section, f"Session template missing metadata field: {field}"
    for field in _SESSION_TEMPLATE_EVIDENCE_FIELDS:
        assert field in section, f"Session template missing evidence field: {field}"
    assert section.count("PASS/FRICTION/FAIL/NOT_RUN") >= 8
    for col in ("Severity", "Finding", "Evidence", "Resolution", "Rerun required"):
        assert col in section, f"Session template findings missing: {col}"


def test_outreach_invitation_contract(outreach_text: str) -> None:
    for heading, placeholders in _INVITATION_SECTIONS:
        section = _extract_h2_section(outreach_text, heading)
        for placeholder in placeholders:
            assert placeholder in section, f"{heading} missing placeholder: {placeholder}"
        assert not _MOVING_REPO_URL.search(section), f"{heading} contains moving repo URL"
        assert not _RELATIVE_PARTICIPANT_LINK.search(section), f"{heading} contains relative link"
    maintainer = outreach_text.split("## Blind first-contact invitation", 1)[0]
    assert _RELATIVE_PARTICIPANT_LINK.search(maintainer), "Maintainer sections should retain relative links"


def test_no_internal_task_ids_in_public_docs(
    protocol_text: str,
    outreach_text: str,
    launch_text: str,
    evaluation_text: str,
) -> None:
    for path_name, text in (
        (PROTOCOL_PATH.name, protocol_text),
        (OUTREACH_KIT_PATH.name, outreach_text),
        (LAUNCH_CHECKLIST_PATH.name, launch_text),
        (EVALUATION_GUIDE_PATH.name, evaluation_text),
    ):
        match = _INTERNAL_TASK_PATTERN.search(text)
        assert match is None, f"{path_name} contains internal task ID: {match.group() if match else ''}"


def test_architecture_layer_aware_routing() -> None:
    text = _read(PUBLIC_ARCHITECTURE_PATH)
    assert "reader-facing Layer 1" in text
    assert "Layer 5 maintainer control" in text
    assert "docs/project/maintainers/public-adoption/README.md" in text
    assert "must not become a normal public-reader route" in text
    public_map = _read(REPO_ROOT / "docs" / "project" / "community" / "PUBLIC_DOCUMENTATION_MAP.md")
    assert "EXTERNAL_READER_VALIDATION_PROTOCOL" not in public_map


def test_architecture_task_id_guard() -> None:
    text = _read(PUBLIC_ARCHITECTURE_PATH)
    assert "PUBLIC-DOCS-COMMERCIALIZATION-9A" in text
    for leaked in (
        "PUBLIC-DOCS-COMMERCIALIZATION-9B",
        "CTX-UCL-",
        "LKW-SLACK-",
        "GOOGLE-WORKSPACE-",
        "MSGRAPH-",
    ):
        assert leaked not in text, f"Architecture leaked task ID: {leaked}"


def test_brevity() -> None:
    limits = {
        PROTOCOL_PATH: 400,
        EVALUATION_GUIDE_PATH: 280,
        LAUNCH_CHECKLIST_PATH: 220,
        OUTREACH_KIT_PATH: 320,
    }
    for path, max_lines in limits.items():
        count = len(_read(path).splitlines())
        assert count <= max_lines, f"{path.name} has {count} lines (max {max_lines})"


def test_reader_documents_remain_untouched() -> None:
    for link in ("README.md", "COLLABORATION.md", "LICENSE"):
        assert link in _read(README_PATH)


_POSITIVE_FORBIDDEN_CLAIM_MUTATIONS = (
    "Intergrax is commercially validated.",
    "Intergrax is production-ready.",
    "Intergrax is production ready.",
    "External validation is complete.",
    "Real-user validation is complete.",
    "Product validated.",
    "Usability certified.",
    "Intergrax remains commercially validated.",
    "Without doubt, Intergrax is production-ready.",
    "There is no ambiguity: external validation is complete.",
    "No reviewer objected, and Intergrax is product validated.",
    "The project remains production ready.",
)

_LEGITIMATE_NEGATIVE_CLAIM_MUTATIONS = (
    "Commercial validation is incomplete.",
    "Intergrax is not production-ready.",
    "Intergrax is not production ready.",
    "External validation is not complete.",
    "Real-user validation remains incomplete.",
    "Creating this protocol does not mean external validation is complete.",
    "A protocol does not constitute external validation.",
    "The review proceeds without claiming production readiness.",
    "No completed external validation is claimed.",
)

_MIXED_FORBIDDEN_CLAIM_MUTATIONS = (
    "External validation is not complete. Intergrax is production-ready.",
    "- Commercial validation is incomplete, but Intergrax is production-ready.",
)

_MIXED_ALLOWED_CLAIM_MUTATIONS = (
    "| Production status | Intergrax is not production-ready |",
)


@pytest.mark.parametrize("text", _POSITIVE_FORBIDDEN_CLAIM_MUTATIONS)
def test_positive_forbidden_claim_mutations_raise(text: str) -> None:
    with pytest.raises(AssertionError):
        _assert_no_positive_forbidden_claim(text, "mutation")


@pytest.mark.parametrize("text", _LEGITIMATE_NEGATIVE_CLAIM_MUTATIONS)
def test_legitimate_negative_claim_mutations_pass(text: str) -> None:
    _assert_no_positive_forbidden_claim(text, "mutation")


@pytest.mark.parametrize("text", _MIXED_FORBIDDEN_CLAIM_MUTATIONS)
def test_mixed_forbidden_claim_mutations_raise(text: str) -> None:
    with pytest.raises(AssertionError):
        _assert_no_positive_forbidden_claim(text, "mutation")


@pytest.mark.parametrize("text", _MIXED_ALLOWED_CLAIM_MUTATIONS)
def test_mixed_allowed_claim_mutations_pass(text: str) -> None:
    _assert_no_positive_forbidden_claim(text, "mutation")
