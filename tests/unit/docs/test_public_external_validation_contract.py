# © Artur Czarnecki. All rights reserved.

"""PUBLIC-DOCS-COMMERCIALIZATION-9A: external reader validation contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_GUIDE_PATH = REPO_ROOT / "EVALUATION_GUIDE.md"
PROTOCOL_PATH = REPO_ROOT / "docs" / "public-adoption" / "EXTERNAL_READER_VALIDATION_PROTOCOL.md"
LAUNCH_CHECKLIST_PATH = REPO_ROOT / "docs" / "public-adoption" / "PUBLIC_LAUNCH_CHECKLIST.md"
OUTREACH_KIT_PATH = REPO_ROOT / "docs" / "public-adoption" / "OUTREACH_KIT.md"
PUBLIC_ADOPTION_INDEX_PATH = REPO_ROOT / "docs" / "public-adoption" / "README.md"
PUBLIC_ARCHITECTURE_PATH = REPO_ROOT / "docs" / "public-adoption" / "PUBLIC_DOCUMENTATION_ARCHITECTURE.md"
README_PATH = REPO_ROOT / "README.md"
FAQ_PATH = REPO_ROOT / "FAQ.md"
WHY_PATH = REPO_ROOT / "WHY_INTERGRAX.md"
USE_CASES_PATH = REPO_ROOT / "USE_CASES.md"
ROADMAP_PATH = REPO_ROOT / "ROADMAP.md"
BUILD_PATH = REPO_ROOT / "BUILD_WITH_INTERGRAX.md"
PROOFS_PATH = REPO_ROOT / "PROOFS.md"
PARTNERS_PATH = REPO_ROOT / "PARTNERS.md"
COLLABORATION_PATH = REPO_ROOT / "COLLABORATION.md"
LICENSE_PATH = REPO_ROOT / "LICENSE"
LKW_PROOF_PATH = REPO_ROOT / "docs" / "public-adoption" / "LKW_PLATFORM_PROOF.md"
TOKEN_GUIDE_PATH = REPO_ROOT / "docs" / "features" / "token_optimization" / "README.md"

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

_FORBIDDEN_COMPLETION_CLAIMS = (
    "external validation is complete",
    "validated by external users",
    "real-user validation is complete",
    "commercial validation is complete",
    "commercially validated",
    "product validated",
    "production ready",
    "usability certified",
)

_NEGATION_MARKERS = (
    "not ",
    "does not",
    "do not",
    "no ",
    "incomplete",
    "remain",
    "without",
    "does not mean",
    "not constitute",
    "not a ",
    "not complete",
    "not_started",
    "not started",
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
    "docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md",
)

_ISSUE_URL_PATTERN = re.compile(
    r"github\.com/jakbuczarnecki/intergrax/issues/\d+",
    re.IGNORECASE,
)

_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

_INTERNAL_TASK_PATTERN = re.compile(
    r"(CTX-UCL-|TOKEN-10|LKW-SLACK-|GOOGLE-WORKSPACE-KNOWLEDGE-|"
    r"MSGRAPH-KNOWLEDGE-|PUBLIC-DOCS-COMMERCIALIZATION-)",
    re.IGNORECASE,
)

_EVAL_GUIDE_LINKS = (
    "README.md",
    "FAQ.md",
    "WHY_INTERGRAX.md",
    "USE_CASES.md",
    "ROADMAP.md",
    "BUILD_WITH_INTERGRAX.md",
    "PROOFS.md",
    "ARCHITECTURE_OVERVIEW.md",
    "PARTNERS.md",
    "COLLABORATION.md",
    "LICENSE",
    "docs/PUBLIC_DOCUMENTATION_MAP.md",
    "docs/DOCUMENTATION_MAP.md",
    "docs/public-adoption/LKW_PLATFORM_PROOF.md",
    "docs/features/token_optimization/README.md",
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


def _through_at_a_glance(text: str) -> str:
    at_glance = re.search(r"^## At a glance\s*$", text, re.MULTILINE)
    if not at_glance:
        raise AssertionError("Missing ## At a glance section")
    after = text[at_glance.end() :]
    next_h2 = re.search(r"^## ", after, re.MULTILINE)
    if not next_h2:
        raise AssertionError("Missing H2 section after ## At a glance")
    return text[: at_glance.end() + next_h2.start()]


def _assert_no_positive_forbidden_claim(text: str, path_name: str) -> None:
    lower = _normalize(text)
    for phrase in _FORBIDDEN_COMPLETION_CLAIMS:
        start = 0
        while True:
            idx = lower.find(phrase, start)
            if idx == -1:
                break
            context = lower[max(0, idx - 80) : idx + len(phrase) + 80]
            assert any(marker in context for marker in _NEGATION_MARKERS), (
                f"{path_name}: positive forbidden claim {phrase!r} at index {idx}"
            )
            start = idx + 1


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
    opening = evaluation_text[: evaluation_text.find("## Who this guide is for")]
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
        "Repository baseline",
        "Reader journey readiness",
        "Validation protocol readiness",
        "Claims and legal boundaries",
        "Outreach readiness",
        "Evidence and privacy",
        "Final gate",
    ):
        assert section in launch_text, f"Launch checklist missing section: {section}"
    assert "READY_TO_RUN" in launch_text
    assert "BLOCKED" in launch_text


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


def test_no_internal_task_ids_in_evaluation_guide(evaluation_text: str) -> None:
    match = _INTERNAL_TASK_PATTERN.search(evaluation_text)
    assert match is None, f"EVALUATION_GUIDE contains internal task ID: {match.group() if match else ''}"


def test_brevity() -> None:
    limits = {
        PROTOCOL_PATH: 380,
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
