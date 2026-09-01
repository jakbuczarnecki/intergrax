# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q1 gate over real RAG/C1 qualification report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.external_proof,
    pytest.mark.qualification,
    pytest.mark.no_ci,
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPORT_PATH = _REPO_ROOT / ".tmp" / "session" / "diag-functional-q1" / "qualification-report.json"


def _services_configured() -> bool:
    return bool(os.environ.get("LKW_BASE_URL"))


@pytest.mark.skipif(not _services_configured(), reason="LKW_BASE_URL not configured for Q1 qualification")
def test_diag_functional_q1_real_rag_qualification_passes() -> None:
    from tests.system.functional_diagnostics_q1.runner import run_qualification

    report = run_qualification()
    assert report.verdict in {"PASS", "FAILED", "BLOCKED"}
    if report.verdict == "BLOCKED":
        pytest.fail(report.blocked_reason or "Q1 qualification blocked")
    assert report.verdict == "PASS"
    assert report.false_positive_cases == 0
    assert report.false_negative_cases == 0
    assert report.repeatability_pass is True


def test_diag_functional_q1_report_schema_when_present() -> None:
    if not _REPORT_PATH.is_file():
        pytest.skip("Q1 qualification report not present")
    payload = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    assert payload["verdict"] in {"PASS", "FAILED", "BLOCKED"}
    assert "matched_cases" in payload
    assert "records" in payload
