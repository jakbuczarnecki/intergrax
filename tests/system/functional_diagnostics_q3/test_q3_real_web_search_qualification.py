# © Artur Czarnecki. All rights reserved.

"""DIAG-FUNCTIONAL-Q3 gate over real web-search qualification report."""

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
_REPORT_PATH = _REPO_ROOT / ".tmp" / "session" / "diag-functional-q3" / "qualification-report.json"


def _services_configured() -> bool:
    return bool(os.environ.get("LKW_BASE_URL"))


@pytest.mark.skipif(not _services_configured(), reason="LKW_BASE_URL not configured for Q3 qualification")
def test_diag_functional_q3_real_web_search_qualification_passes() -> None:
    from tests.system.functional_diagnostics_q3.runner import run_qualification

    report = run_qualification()
    assert report.verdict in {"PASS", "FAILED", "BLOCKED"}
    if report.verdict == "BLOCKED":
        pytest.fail(report.blocked_reason or "Q3 qualification blocked")
    assert report.verdict == "PASS"
    assert report.false_positive_cases == 0
    assert report.false_negative_cases == 0
    assert report.repeatability_pass is True


def test_diag_functional_q3_report_artifact_exists_after_run() -> None:
    if not _REPORT_PATH.is_file():
        pytest.skip("Run Q3 qualification first")
    payload = json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    assert payload.get("verdict") in {"PASS", "FAILED", "BLOCKED"}
