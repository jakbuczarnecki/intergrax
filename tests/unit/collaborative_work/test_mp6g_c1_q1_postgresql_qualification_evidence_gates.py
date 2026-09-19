# © Artur Czarnecki. All rights reserved.

"""MP-6G-C1-Q1 — documentation gates for live PostgreSQL E2E qualification evidence."""

from __future__ import annotations

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
    / "MP-6G_E2E_ISOLATION_IDEMPOTENCY_QUALIFICATION.md"
)

_QUALIFICATION_SHA = "5ff90667569bf23c88a4db98d489966e309f4ab7"
_MP6G_C1_HARNESS_SHA = "30900bbd566d956fb5df7f16b00e15057a5a7a91"

_REQUIRED_MARKERS = (
    "MP-6G",
    "MP-6G — QUALIFIED / CERTIFIED",
    "baseline_sha",
    "31ef13456aff1753313a4dcd5c0933f79be0561a",
    "initial_mp6g_harness_sha",
    "fefd4bf9b2317ebe0f451c37fce8b1db1972203f",
    "mp6g_c1_harness_sha",
    _MP6G_C1_HARNESS_SHA,
    "qualification_sha",
    _QUALIFICATION_SHA,
    "PostgreSQL",
    "source_tree_mode",
    "clean exact committed checkout",
    "run_mp6g_e2e_contract_suite",
    "passed",
    "skipped",
    "xfailed",
    "failed",
    "**skipped** | 0",
    "**xfailed** | 0",
    "**failed** | 0",
    "WorkItem",
    "Assignment",
    "WorkArtifact",
    "Decision",
    "ContextView",
    "WORK_ITEM_UPDATED — NOT APPLICABLE",
    "PublicationPort",
    "BLOCKING FINDINGS: NONE",
    "MP-6G-C1-Q1",
    "CLOSED",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _canonical_status_section(text: str) -> str:
    start = text.find("## Canonical status")
    if start < 0:
        return ""
    end = text.find("\n## ", start + 1)
    return text[start:end] if end > 0 else text[start:]


def test_mp6g_c1_q1_postgresql_qualification_evidence_artifact_exists() -> None:
    assert _EVIDENCE.is_file(), "missing MP-6G PostgreSQL qualification evidence artifact"


def test_mp6g_c1_q1_postgresql_qualification_evidence_markers() -> None:
    text = _read(_EVIDENCE)
    missing = [marker for marker in _REQUIRED_MARKERS if marker not in text]
    assert not missing, f"MP-6G-C1-Q1 evidence: missing markers: {missing}"


def test_mp6g_c1_q1_canonical_status_not_pending_postgresql() -> None:
    text = _canonical_status_section(_read(_EVIDENCE))
    assert text, "canonical status section missing"
    assert "POSTGRESQL QUALIFICATION PENDING" not in text, (
        "canonical status must not remain POSTGRESQL QUALIFICATION PENDING after closure"
    )
