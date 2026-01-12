# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unit tests for LLMAdapterUsageLog.

These tests define the behavioral contract for per-adapter usage tracking:
- begin_call()/end_call() update aggregated stats deterministically,
- success vs failure affects error counters,
- get_run_stats(run_id) filters correctly,
- unknown run_id returns an empty/zero stats object,
- duration_ms is always non-negative (no clock regressions).

Why this matters:
Usage telemetry is production-critical for cost control and observability.
Regressions here cause incorrect billing, broken dashboards, and misleading diagnostics.
"""

from __future__ import annotations

import pytest

from intergrax.llm_adapters.llm_adapter import LLMAdapterUsageLog


pytestmark = pytest.mark.unit


def test_initial_stats_are_zero() -> None:
    """
    A fresh usage log must start with zero counts and tokens.
    """
    usage = LLMAdapterUsageLog()

    total = usage.get_run_stats(None)
    assert total.calls == 0
    assert total.errors == 0
    assert total.input_tokens == 0
    assert total.output_tokens == 0
    assert total.total_tokens == 0


def test_successful_call_updates_tokens_and_calls() -> None:
    """
    A successful call must increment calls and token counters (no errors).
    """
    usage = LLMAdapterUsageLog()
    run_id = "run-001"

    call = usage.begin_call(run_id=run_id)
    usage.end_call(call, input_tokens=10, output_tokens=5, success=True, error_type=None)

    stats = usage.get_run_stats(run_id)
    assert stats.calls == 1
    assert stats.errors == 0
    assert stats.input_tokens == 10
    assert stats.output_tokens == 5
    assert stats.total_tokens == 15

    # duration should be present and non-negative (do not assert exact value)
    assert stats.duration_ms >= 0


def test_failed_call_increments_errors_and_tokens() -> None:
    """
    A failed call must increment errors and still account for tokens if provided.
    """
    usage = LLMAdapterUsageLog()
    run_id = "run-001"

    call = usage.begin_call(run_id=run_id)
    usage.end_call(call, input_tokens=3, output_tokens=1, success=False, error_type="timeout")

    stats = usage.get_run_stats(run_id)
    assert stats.calls == 1
    assert stats.errors == 1
    assert stats.input_tokens == 3
    assert stats.output_tokens == 1
    assert stats.total_tokens == 4
    assert stats.duration_ms >= 0


def test_stats_are_filtered_by_run_id() -> None:
    """
    get_run_stats(run_id) must return only calls belonging to that run_id.
    """
    usage = LLMAdapterUsageLog()

    # run-1: 1 success
    c1 = usage.begin_call(run_id="run-1")
    usage.end_call(c1, input_tokens=2, output_tokens=3, success=True, error_type=None)

    # run-2: 1 failure
    c2 = usage.begin_call(run_id="run-2")
    usage.end_call(c2, input_tokens=5, output_tokens=0, success=False, error_type="error")

    s1 = usage.get_run_stats("run-1")
    assert s1.calls == 1
    assert s1.errors == 0
    assert s1.total_tokens == 5

    s2 = usage.get_run_stats("run-2")
    assert s2.calls == 1
    assert s2.errors == 1
    assert s2.total_tokens == 5

    general = usage.get_run_stats(None)
    assert general.calls == 0
    assert general.errors == 0
    assert general.total_tokens == 0


def test_unknown_run_id_returns_zero_stats() -> None:
    """
    Unknown run_id must return a zero/empty stats object (no KeyError, no None).
    """
    usage = LLMAdapterUsageLog()

    stats = usage.get_run_stats("missing-run")
    assert stats.calls == 0
    assert stats.errors == 0
    assert stats.total_tokens == 0
    assert stats.duration_ms == 0
