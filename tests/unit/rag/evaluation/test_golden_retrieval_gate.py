# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.rag.evaluation.golden_harness import load_golden_cases, run_golden_retrieval

pytestmark = [pytest.mark.unit, pytest.mark.gate]

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "rag_golden" / "retrieval_cases.json"


def test_golden_retrieval_regression() -> None:
    cases = load_golden_cases(FIXTURE)
    report = run_golden_retrieval(cases)
    assert report.passed, [f"{r.name}: recall={r.recall}" for r in report.results if not r.passed]
