# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-2A: regression benchmark report artifact tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_kind_registry import clear_event_kind_registry
from intergrax.runtime.token_optimization.domain_events import (
    register_token_optimization_domain_signal,
)
from intergrax.runtime.token_optimization.emission import (
    TokenOptimizationEmissionPolicy,
    TokenOptimizationEmissionStatus,
)
from intergrax.runtime.token_optimization.regression import run_token_regression_benchmarks
from intergrax.runtime.token_optimization.regression_emission import (
    run_token_regression_benchmarks_with_emission,
)
from intergrax.runtime.token_optimization.regression_report import (
    build_token_regression_report,
    format_token_regression_report,
    token_regression_report_to_dict,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_REGRESSION_MODULE = (
    _REPO_ROOT / "intergrax" / "runtime" / "token_optimization" / "regression.py"
)
_EMISSION_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "token_optimization"
    / "regression_emission.py"
)
_BENCHMARK_SCRIPT = _REPO_ROOT / "scripts" / "check_token_regression_benchmarks.py"
_SCOPE_GUARD_MODULES = (_REGRESSION_MODULE, _EMISSION_MODULE)
_SCOPE_GUARD_IMPORT = "regression_report"

_UNSAFE_KEYS = frozenset(
    {
        "content",
        "raw_content",
        "original_content",
        "optimized_content",
        "prompt",
        "messages",
        "document",
        "documents",
        "memory",
        "memory_content",
        "summary_text",
        "tool_schema",
        "tool_catalog",
        "context",
        "context_pack",
        "fragments",
        "evidence",
        "payload",
        "body",
        "raw_context",
        "raw_prompt",
        "raw_document",
        "tool_args",
        "chunks",
    }
)


@pytest.fixture(autouse=True)
def _register_token_optimization_domain_kind() -> None:
    clear_event_kind_registry()
    register_token_optimization_domain_signal()
    yield
    clear_event_kind_registry()


def _emit_context() -> EmitContext:
    return EmitContext(
        task_id="task-1",
        run_id="run-1",
        tenant_id="tenant-a",
        bus=RuntimeEventBus(record_history=True),
    )


def _enabled_emission_policy() -> TokenOptimizationEmissionPolicy:
    return TokenOptimizationEmissionPolicy(enabled=True)


def _collect_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_collect_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_collect_keys(nested))
    return keys


def test_builds_report_from_default_regression_run() -> None:
    summary = run_token_regression_benchmarks()
    report = build_token_regression_report(
        summary,
        report_id="report-test-1",
        generated_at="2026-07-02T12:00:00+00:00",
    )

    assert report.report_id == "report-test-1"
    assert report.generated_at == "2026-07-02T12:00:00+00:00"
    assert len(report.results) == summary.total_fixtures
    assert report.emission is None


def test_report_totals_match_summary() -> None:
    summary = run_token_regression_benchmarks()
    report = build_token_regression_report(
        summary,
        report_id="report-test-2",
        generated_at="2026-07-02T12:00:00+00:00",
    )

    assert report.total_fixtures == summary.total_fixtures
    assert report.passed == summary.passed
    assert report.failed == summary.failed
    assert report.total_baseline_tokens == summary.total_baseline_tokens
    assert report.total_optimized_tokens == summary.total_optimized_tokens
    assert report.total_saved_tokens == summary.total_saved_tokens
    assert report.total_saved_ratio == pytest.approx(summary.total_saved_ratio)


def test_report_items_preserve_fixture_ids_and_token_savings_fields() -> None:
    summary = run_token_regression_benchmarks()
    report = build_token_regression_report(
        summary,
        report_id="report-test-3",
        generated_at="2026-07-02T12:00:00+00:00",
    )

    for item, result in zip(report.results, summary.results, strict=True):
        assert item.fixture_id == result.fixture_id
        assert item.baseline_tokens == result.baseline_tokens
        assert item.optimized_tokens == result.optimized_tokens
        assert item.saved_tokens == result.saved_tokens
        assert item.saved_ratio == pytest.approx(result.saved_ratio)
        assert item.token_category is not None


def test_report_breakdowns_separate_eval_case_metrics() -> None:
    summary = run_token_regression_benchmarks()
    report = build_token_regression_report(
        summary,
        report_id="report-test-breakdowns",
        generated_at="2026-07-02T12:00:00+00:00",
    )

    breakdowns = {breakdown.eval_case: breakdown for breakdown in report.breakdowns}
    assert set(breakdowns) == {"compactable", "protected", "fallback"}

    compactable = breakdowns["compactable"]
    compactable_items = [item for item in report.results if item.eval_case == "compactable"]
    assert compactable.total_fixtures == len(compactable_items)
    assert compactable.baseline_tokens == sum(item.baseline_tokens for item in compactable_items)
    assert compactable.saved_tokens == sum(item.saved_tokens for item in compactable_items)
    assert compactable.saved_ratio == pytest.approx(
        compactable.saved_tokens / compactable.baseline_tokens
    )

    protected = breakdowns["protected"]
    assert protected.total_fixtures == 3
    assert protected.noop_count == 3
    assert protected.fallback_count == 0
    assert protected.saved_tokens == 0

    fallback = breakdowns["fallback"]
    assert fallback.total_fixtures == 1
    assert fallback.noop_count == 1
    assert fallback.fallback_count == 1
    assert fallback.saved_tokens == 0


def test_dict_conversion_is_json_serializable() -> None:
    summary = run_token_regression_benchmarks()
    report = build_token_regression_report(
        summary,
        report_id="report-test-4",
        generated_at="2026-07-02T12:00:00+00:00",
    )

    payload = token_regression_report_to_dict(report)
    serialized = json.dumps(payload, sort_keys=True)
    assert json.loads(serialized) == payload
    assert {row["eval_case"] for row in payload["breakdowns"]} == {
        "compactable",
        "protected",
        "fallback",
    }


def test_human_format_contains_totals_breakdowns_and_per_fixture_lines() -> None:
    summary = run_token_regression_benchmarks()
    report = build_token_regression_report(
        summary,
        report_id="report-test-5",
        generated_at="2026-07-02T12:00:00+00:00",
    )

    formatted = format_token_regression_report(report)

    assert "fixtures=7 passed=7 failed=0" in formatted
    assert f"baseline={report.total_baseline_tokens}" in formatted
    assert "breakdown:" in formatted
    assert "compactable fixtures=3" in formatted
    assert "protected fixtures=3" in formatted
    assert "fallback fixtures=1" in formatted
    for item in report.results:
        assert item.fixture_id in formatted
        assert f"saved={item.saved_tokens}" in formatted


def test_report_items_include_safe_eval_metadata() -> None:
    summary = run_token_regression_benchmarks()
    report = build_token_regression_report(
        summary,
        report_id="report-test-eval",
        generated_at="2026-07-02T12:00:00+00:00",
    )

    for item in report.results:
        assert item.eval_case in {"compactable", "protected", "fallback"}
        assert item.expected_behavior
        assert item.expectation_status in {"met", "failed"}

    payload = token_regression_report_to_dict(report)
    for row in payload["results"]:
        assert "eval_case" in row
        assert "expected_behavior" in row
        assert "expectation_status" in row
        assert "content" not in row
        assert "original_content" not in row


def test_emission_report_includes_counts_and_statuses() -> None:
    ctx = _emit_context()
    emission_run = run_token_regression_benchmarks_with_emission(
        ctx,
        emission_policy=_enabled_emission_policy(),
    )
    report = build_token_regression_report(
        emission_run.summary,
        emission_run=emission_run,
        report_id="report-test-6",
        generated_at="2026-07-02T12:00:00+00:00",
    )

    assert report.emission is not None
    assert report.emission.attempted_result_emissions == len(emission_run.summary.results)
    assert report.emission.summary_emission_attempted is True
    assert report.emission.emitted_event_count == emission_run.emitted_event_count
    assert report.emission.emitted == len(emission_run.summary.results) + 1
    assert report.emission.skipped_disabled == 0
    assert report.emission.skipped_kind_disabled == 0
    assert report.emission.dry_run == 0


def test_disabled_emission_report_shows_zero_emitted_and_skipped_statuses() -> None:
    ctx = _emit_context()
    emission_run = run_token_regression_benchmarks_with_emission(ctx)
    report = build_token_regression_report(
        emission_run.summary,
        emission_run=emission_run,
        report_id="report-test-7",
        generated_at="2026-07-02T12:00:00+00:00",
    )

    assert report.emission is not None
    assert report.emission.emitted_event_count == 0
    assert report.emission.emitted == 0
    assert report.emission.skipped_disabled == len(emission_run.summary.results) + 1
    assert report.emission.skipped_kind_disabled == 0
    assert report.emission.dry_run == 0
    assert all(
        emission.status == TokenOptimizationEmissionStatus.SKIPPED_DISABLED
        for emission in emission_run.result_emissions
    )


def test_enabled_emission_report_shows_len_results_plus_one_emitted_events() -> None:
    ctx = _emit_context()
    emission_run = run_token_regression_benchmarks_with_emission(
        ctx,
        emission_policy=_enabled_emission_policy(),
    )
    report = build_token_regression_report(
        emission_run.summary,
        emission_run=emission_run,
        report_id="report-test-8",
        generated_at="2026-07-02T12:00:00+00:00",
    )

    expected = len(emission_run.summary.results) + 1
    assert report.emission is not None
    assert report.emission.emitted_event_count == expected
    assert report.emission.emitted == expected


def test_dry_run_emission_report_shows_zero_emitted_and_dry_run_statuses() -> None:
    ctx = _emit_context()
    policy = TokenOptimizationEmissionPolicy(enabled=True, dry_run=True)
    emission_run = run_token_regression_benchmarks_with_emission(
        ctx,
        emission_policy=policy,
    )
    report = build_token_regression_report(
        emission_run.summary,
        emission_run=emission_run,
        report_id="report-test-9",
        generated_at="2026-07-02T12:00:00+00:00",
    )

    assert report.emission is not None
    assert report.emission.emitted_event_count == 0
    assert report.emission.emitted == 0
    assert report.emission.dry_run == len(emission_run.summary.results) + 1
    assert report.emission.skipped_disabled == 0


def test_unsafe_metadata_and_raw_content_keys_are_not_present() -> None:
    summary = run_token_regression_benchmarks()
    report = build_token_regression_report(
        summary,
        report_id="report-test-10",
        generated_at="2026-07-02T12:00:00+00:00",
        metadata={
            "run_id": "run-safe-1",
            "content": "secret prompt body",
            "original_content": "raw original",
            "optimized_content": "raw optimized",
            "payload": {"nested": "value"},
        },
    )

    payload = token_regression_report_to_dict(report)
    formatted = format_token_regression_report(report)

    assert _collect_keys(payload).isdisjoint(_UNSAFE_KEYS)
    assert "secret prompt body" not in formatted
    assert "raw original" not in formatted
    assert "raw optimized" not in formatted
    assert payload["metadata"]["run_id"] == "run-safe-1"
    assert "content" not in payload["metadata"]
    assert "original_content" not in payload["metadata"]
    assert "optimized_content" not in payload["metadata"]
    assert "payload" not in payload["metadata"]


def test_report_is_deterministic_when_id_and_timestamp_are_provided() -> None:
    summary = run_token_regression_benchmarks()
    kwargs = {
        "report_id": "report-deterministic",
        "generated_at": "2026-07-02T12:00:00+00:00",
        "metadata": {"run_id": "run-deterministic"},
    }

    first = build_token_regression_report(summary, **kwargs)
    second = build_token_regression_report(summary, **kwargs)

    assert token_regression_report_to_dict(first) == token_regression_report_to_dict(second)
    assert format_token_regression_report(first) == format_token_regression_report(second)


@pytest.mark.parametrize("module_path", _SCOPE_GUARD_MODULES, ids=lambda p: p.name)
def test_core_modules_and_benchmark_script_do_not_import_regression_report(
    module_path: Path,
) -> None:
    source = module_path.read_text(encoding="utf-8")
    assert _SCOPE_GUARD_IMPORT not in source
