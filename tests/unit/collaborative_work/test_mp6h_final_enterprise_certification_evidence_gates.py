# © Artur Czarnecki. All rights reserved.

"""MP-6H — final enterprise certification evidence gates."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_EVIDENCE = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "MP-6_FINAL_ENTERPRISE_CERTIFICATION.md"
)

_AUDITED_SHA = "5c32dd68a9f3da2ccc21f6e8cffec2522450a60d"
_QUALIFICATION_SHA = "5ff90667569bf23c88a4db98d489966e309f4ab7"
_PROVENANCE_CORRECTION_SHA = "5c32dd68a9f3da2ccc21f6e8cffec2522450a60d"

_STATUS_DOCS = {
    "collaborative_work_architecture": _REPO
    / "docs"
    / "project"
    / "architecture"
    / "COLLABORATIVE_WORK.md",
    "collaborative_work_plan": _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "plans"
    / "COLLABORATIVE_WORK.md",
    "multiplayer_architecture": _REPO
    / "docs"
    / "project"
    / "capabilities"
    / "architecture"
    / "MULTIPLAYER_AI.md",
    "multiplayer_plan": _REPO
    / "docs"
    / "project"
    / "capabilities"
    / "plan"
    / "MULTIPLAYER_AI.md",
}

_REQUIRED_MARKERS = (
    "MP-6H",
    "MP-6 — ENTERPRISE CERTIFIED / CLOSED",
    "MP-6H — CLOSED / CERTIFIED",
    "audited_sha",
    _AUDITED_SHA,
    "qualification_sha",
    _QUALIFICATION_SHA,
    _PROVENANCE_CORRECTION_SHA,
    "MP-6D-Q1_POSTGRESQL_PROVIDER_QUALIFICATION.md",
    "MP-6G_E2E_ISOLATION_IDEMPOTENCY_QUALIFICATION.md",
    "BLOCKING FINDINGS: NONE",
    "Known limitations",
    "WORK_ITEM_UPDATED",
    "no global total ordering",
    "forward live-feed",
    "cross-store atomic",
    "MP-6A",
    "MP-6B",
    "MP-6C",
    "MP-6D",
    "MP-6E",
    "MP-6F",
    "MP-6G",
    "MP-6H",
    "Contract matrix",
    "Security matrix",
    "Persistence matrix",
    "Source integration matrix",
)

_STALE_PATTERNS = (
    re.compile(r"MP-6\s+—\s+IN PROGRESS", re.I),
    re.compile(r"MP-6H\s+—\s+NEXT", re.I),
    re.compile(r"MP-6G\s+PG\s+PENDING", re.I),
    re.compile(r"MP-6F\s+—\s+NEXT", re.I),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _canonical_status_section(text: str) -> str:
    start = text.find("## 1. Wynik")
    if start < 0:
        return text[:4000]
    end = text.find("## 20. Independent audit requirement")
    if end < 0:
        return text[start:]
    return text[start:end]


def test_mp6h_final_certification_artifact_exists() -> None:
    assert _EVIDENCE.is_file()


def test_mp6h_evidence_required_markers() -> None:
    text = _read(_EVIDENCE)
    missing = [m for m in _REQUIRED_MARKERS if m not in text]
    assert not missing, missing


def test_mp6h_evidence_no_stale_status_in_canonical_sections() -> None:
    text = _canonical_status_section(_read(_EVIDENCE))
    for pattern in _STALE_PATTERNS:
        match = pattern.search(text)
        assert match is None, f"stale status in certification artifact: {match.group(0)!r}"


def test_mp6h_closure_markers_in_ssot_docs() -> None:
    for name, path in _STATUS_DOCS.items():
        text = _read(path)
        assert "MP-6 — ENTERPRISE CERTIFIED / CLOSED" in text, name
        assert "MP-6H — CLOSED / CERTIFIED" in text, name


def test_mp6h_ssot_docs_reject_stale_mp6_active_task() -> None:
    for name, path in _STATUS_DOCS.items():
        text = _read(path)
        assert "MP-6H — NEXT" not in text, f"{name}: MP-6H still NEXT"
        assert "Current active task:** **MP-6" not in text, f"{name}: stale MP-6 active task"
        assert "active slice **MP-6F — NEXT**" not in text, f"{name}: stale MP-6F active slice"
