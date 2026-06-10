# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.rag.evaluation.golden_harness import load_golden_cases
from intergrax.rag.evaluation.load_soak import (
    LoadSoakConfig,
    build_soak_retrieval_service,
    run_retrieval_load_soak,
    soak_queries_from_golden_cases,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "rag_golden" / "retrieval_cases.json"


def test_retrieval_load_soak_passes_on_golden_corpus() -> None:
    cases = load_golden_cases(FIXTURE)
    queries = soak_queries_from_golden_cases(cases)
    service = build_soak_retrieval_service(cases)

    result = run_retrieval_load_soak(
        service,
        queries,
        config=LoadSoakConfig(
            concurrent_workers=4,
            queries_per_worker=2,
            max_p95_latency_ms=2_000.0,
        ),
    )

    assert result.passed is True
    assert result.reason == "ok"
    assert result.queries_executed == 8
    assert result.min_observed_recall >= 1.0


def test_retrieval_load_soak_fails_on_latency_budget() -> None:
    cases = load_golden_cases(FIXTURE)
    queries = soak_queries_from_golden_cases(cases)
    service = build_soak_retrieval_service(cases)

    result = run_retrieval_load_soak(
        service,
        queries,
        config=LoadSoakConfig(
            concurrent_workers=2,
            queries_per_worker=1,
            max_p95_latency_ms=0.001,
        ),
    )

    assert result.passed is False
    assert "slo_latency_exceeded" in result.reason
