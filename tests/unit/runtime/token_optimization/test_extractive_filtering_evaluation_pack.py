# © Artur Czarnecki. All rights reserved.

"""TOKEN-OPT-4B: ExtractiveFilteringLayer evaluation / regression pack tests."""

from __future__ import annotations

import re

import pytest

from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationLayerDecision,
    TokenOptimizationSourceType,
)
from intergrax.runtime.token_optimization.layers.extractive_filtering import (
    ExtractiveFilteringLayer,
)
from tests.fixtures.token_optimization.extractive_filtering_corpus import (
    EXTRACTIVE_FILTERING_CORPUS,
    EXTRACTIVE_FILTERING_SYNTHETIC_CORPUS_MARKER,
    STRATEGY_EXTRACTIVE_FILTERING,
    STRATEGY_FALLBACK,
    STRATEGY_NO_OP,
    _ALLOWED_SOURCE_TYPES,
    _ALLOWED_STRATEGIES,
    _TOKEN_NAMED_METRIC_FIELDS,
    build_safe_evaluation_report,
    collect_metric_field_names,
    corpus_contains_forbidden_secret_patterns,
    evaluate_case,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_CASE_ID_PATTERN = re.compile(r"^extractive_filtering\.[a-z0-9_]+$")

_REQUIRED_CASE_IDS = frozenset(
    {
        "extractive_filtering.terminal_verbose_progress_noise",
        "extractive_filtering.terminal_pytest_failure_with_noise",
        "extractive_filtering.terminal_traceback_inside_long_output",
        "extractive_filtering.terminal_repeated_warnings",
        "extractive_filtering.terminal_protected_value_in_body",
        "extractive_filtering.terminal_short_clean_output",
        "extractive_filtering.tool_output_large_json_like_noise",
        "extractive_filtering.log_output_warning_error_mix",
    }
)


def _case_by_id(case_id: str):
    return next(case for case in EXTRACTIVE_FILTERING_CORPUS if case.case_id == case_id)


def test_corpus_is_non_empty() -> None:
    assert len(EXTRACTIVE_FILTERING_CORPUS) >= 8
    assert _REQUIRED_CASE_IDS.issubset({case.case_id for case in EXTRACTIVE_FILTERING_CORPUS})


def test_case_ids_are_unique_and_stable_looking() -> None:
    case_ids = [case.case_id for case in EXTRACTIVE_FILTERING_CORPUS]
    assert len(case_ids) == len(set(case_ids))
    for case_id in case_ids:
        assert _CASE_ID_PATTERN.match(case_id)


def test_every_case_is_explicitly_synthetic() -> None:
    for case in EXTRACTIVE_FILTERING_CORPUS:
        assert case.synthetic_marker == EXTRACTIVE_FILTERING_SYNTHETIC_CORPUS_MARKER


def test_cases_use_only_tool_terminal_or_log_output_source_types() -> None:
    for case in EXTRACTIVE_FILTERING_CORPUS:
        assert case.source_type in _ALLOWED_SOURCE_TYPES
        assert isinstance(case.source_type, TokenOptimizationSourceType)


def test_expected_primary_strategy_is_extractive_filtering_or_safe_fallback_noop() -> None:
    for case in EXTRACTIVE_FILTERING_CORPUS:
        assert case.expected.expected_primary_strategy in _ALLOWED_STRATEGIES


def test_no_forbidden_secret_like_values_in_corpus() -> None:
    violations = corpus_contains_forbidden_secret_patterns(EXTRACTIVE_FILTERING_CORPUS)
    assert violations == []


def test_evaluation_invokes_extractive_filtering_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    original_optimize = ExtractiveFilteringLayer.optimize

    def _tracking_optimize(self: ExtractiveFilteringLayer, request: object) -> object:
        calls.append(request)
        return original_optimize(self, request)

    monkeypatch.setattr(ExtractiveFilteringLayer, "optimize", _tracking_optimize)
    case = _case_by_id("extractive_filtering.terminal_verbose_progress_noise")
    result = evaluate_case(case)

    assert len(calls) == 1
    assert result.strategy == STRATEGY_EXTRACTIVE_FILTERING
    assert result.decision == TokenOptimizationLayerDecision.APPLY.value


def test_verbose_progress_noise_applies_and_saves_chars() -> None:
    case = _case_by_id("extractive_filtering.terminal_verbose_progress_noise")
    result = evaluate_case(case)

    assert result.decision == TokenOptimizationLayerDecision.APPLY.value
    assert result.strategy == STRATEGY_EXTRACTIVE_FILTERING
    assert result.saved_chars > 0
    assert result.fallback_used is False
    assert result.omitted_line_count > 0


def test_pytest_failure_case_preserves_failure_evidence() -> None:
    case = _case_by_id("extractive_filtering.terminal_pytest_failure_with_noise")
    result = evaluate_case(case)

    assert result.decision == TokenOptimizationLayerDecision.APPLY.value
    assert result.strategy == STRATEGY_EXTRACTIVE_FILTERING
    assert result.saved_chars > 0
    assert result.fallback_used is False
    assert result.failure_evidence_preserved is True
    assert result.important_markers_preserved is True


def test_traceback_case_preserves_traceback_block() -> None:
    case = _case_by_id("extractive_filtering.terminal_traceback_inside_long_output")
    result = evaluate_case(case)

    assert result.decision == TokenOptimizationLayerDecision.APPLY.value
    assert result.traceback_block_count >= 1
    assert result.traceback_preserved is True
    assert result.important_markers_preserved is True
    assert result.saved_chars > 0
    assert result.fallback_used is False


def test_repeated_warnings_case_reports_repeated_line_groups() -> None:
    case = _case_by_id("extractive_filtering.terminal_repeated_warnings")
    result = evaluate_case(case)

    assert result.decision == TokenOptimizationLayerDecision.APPLY.value
    assert result.repeated_line_group_count > 0
    assert result.saved_chars > 0
    assert result.warning_signal_preserved is True
    assert result.fallback_used is False


def test_protected_value_case_falls_back() -> None:
    case = _case_by_id("extractive_filtering.terminal_protected_value_in_body")
    result = evaluate_case(case)

    assert result.decision == TokenOptimizationLayerDecision.FALLBACK.value
    assert result.fallback_used is True
    assert result.strategy == STRATEGY_FALLBACK
    assert result.protected_region_count > 0
    assert result.saved_chars == 0


def test_short_clean_output_bypasses_safely() -> None:
    case = _case_by_id("extractive_filtering.terminal_short_clean_output")
    result = evaluate_case(case)
    report = build_safe_evaluation_report(result)

    assert result.decision == TokenOptimizationLayerDecision.BYPASS.value
    assert result.strategy == STRATEGY_NO_OP
    assert result.saved_chars == 0
    assert result.fallback_used is False
    assert "content" not in report
    assert "output_content" not in report


def test_tool_output_json_like_noise_reports_extractive_filtering_only() -> None:
    case = _case_by_id("extractive_filtering.tool_output_large_json_like_noise")
    result = evaluate_case(case)
    report = build_safe_evaluation_report(result)
    field_names = collect_metric_field_names(report)

    assert result.decision == TokenOptimizationLayerDecision.APPLY.value
    assert result.strategy == STRATEGY_EXTRACTIVE_FILTERING
    assert result.saved_chars > 0
    assert "dedupe_saved_chars" not in field_names
    assert "packing_decisions" not in field_names
    assert "semantic" not in " ".join(sorted(field_names)).lower()


def test_log_output_warning_error_mix_preserves_important_lines() -> None:
    case = _case_by_id("extractive_filtering.log_output_warning_error_mix")
    result = evaluate_case(case)

    assert result.decision == TokenOptimizationLayerDecision.APPLY.value
    assert result.important_line_count > 0
    assert result.important_markers_preserved is True
    assert result.warning_signal_preserved is True
    assert result.saved_chars > 0
    assert result.fallback_used is False


def test_reports_use_char_level_metrics_only() -> None:
    for case in EXTRACTIVE_FILTERING_CORPUS:
        result = evaluate_case(case)
        report = build_safe_evaluation_report(result)
        assert report["budget_unit"] == "chars"
        assert isinstance(report["baseline_chars"], int)
        assert isinstance(report["output_chars"], int)
        assert isinstance(report["saved_chars"], int)


def test_reports_do_not_include_token_named_savings_fields() -> None:
    for case in EXTRACTIVE_FILTERING_CORPUS:
        result = evaluate_case(case)
        report = build_safe_evaluation_report(result)
        field_names = collect_metric_field_names(report)
        assert field_names.isdisjoint(_TOKEN_NAMED_METRIC_FIELDS)
        assert not any("token" in name.lower() for name in field_names if name != "strategy")


def test_reports_do_not_include_raw_full_content() -> None:
    for case in EXTRACTIVE_FILTERING_CORPUS:
        result = evaluate_case(case)
        report = build_safe_evaluation_report(result)
        assert result.raw_content_in_report is False
        assert case.content not in str(report)
        assert "original_content" not in report
        assert "current_content" not in report
        assert "output_content" not in report


def test_strategy_attribution_is_extractive_filtering_only() -> None:
    for case in EXTRACTIVE_FILTERING_CORPUS:
        result = evaluate_case(case)
        assert result.strategy in _ALLOWED_STRATEGIES
        assert result.strategy == case.expected.expected_primary_strategy
        report = build_safe_evaluation_report(result)
        assert report["strategy"] in _ALLOWED_STRATEGIES
        field_blob = " ".join(sorted(collect_metric_field_names(report))).lower()
        assert "deduplication" not in field_blob
        assert "budget_aware_packing" not in field_blob
        assert "cache_prefix" not in field_blob
        assert "truncation" not in field_blob
