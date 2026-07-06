# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-3E-F: stronger optimizer internal evaluation pack tests."""

from __future__ import annotations

import re

import pytest

from intergrax.runtime.token_optimization.contracts import (
    ContextFragmentPriority,
    ContextPackingDecisionKind,
    TokenOptimizationSourceType,
)
from tests.fixtures.token_optimization.stronger_optimizer_corpus import (
    STRONGER_OPTIMIZER_CORPUS,
    SYNTHETIC_CORPUS_MARKER,
    STRATEGY_BUDGET_AWARE_PACKING,
    STRATEGY_DEDUPLICATION,
    STRATEGY_NO_OP,
    _PRIORITY_VALUES,
    _TOKEN_NAMED_METRIC_FIELDS,
    _collect_string_values,
    build_safe_evaluation_report,
    collect_metric_field_names,
    corpus_contains_forbidden_secret_patterns,
    evaluate_case,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_CASE_ID_PATTERN = re.compile(r"^stronger_opt\.[a-z0-9_]+$")


def _case_by_id(case_id: str):
    return next(case for case in STRONGER_OPTIMIZER_CORPUS if case.case_id == case_id)


def _packing_cases():
    return [case for case in STRONGER_OPTIMIZER_CORPUS if case.expected.packing_applicable]


def _dedupe_cases():
    return [case for case in STRONGER_OPTIMIZER_CORPUS if case.expected.dedupe_applicable]


# --- Corpus validation tests ---


def test_corpus_is_non_empty() -> None:
    assert len(STRONGER_OPTIMIZER_CORPUS) >= 8


def test_case_ids_are_unique_and_stable_looking() -> None:
    case_ids = [case.case_id for case in STRONGER_OPTIMIZER_CORPUS]
    assert len(case_ids) == len(set(case_ids))
    for case_id in case_ids:
        assert _CASE_ID_PATTERN.match(case_id)


def test_every_case_has_source_type() -> None:
    for case in STRONGER_OPTIMIZER_CORPUS:
        assert isinstance(case.source_type, TokenOptimizationSourceType)
        assert case.source_type is not TokenOptimizationSourceType.UNKNOWN


def test_every_case_has_expected_behavior() -> None:
    for case in STRONGER_OPTIMIZER_CORPUS:
        assert case.expected.expected_primary_strategy
        assert isinstance(case.expected.dedupe_applicable, bool)
        assert isinstance(case.expected.packing_applicable, bool)


def test_every_case_is_explicitly_synthetic() -> None:
    for case in STRONGER_OPTIMIZER_CORPUS:
        assert SYNTHETIC_CORPUS_MARKER in case.safety_notes


def test_no_forbidden_secret_like_values_in_corpus() -> None:
    violations = corpus_contains_forbidden_secret_patterns(STRONGER_OPTIMIZER_CORPUS)
    assert violations == []


def test_evaluation_result_and_report_exclude_token_named_metric_fields() -> None:
    for case in STRONGER_OPTIMIZER_CORPUS:
        result = evaluate_case(case)
        report = build_safe_evaluation_report(result, case)
        field_names = collect_metric_field_names(report)
        field_names.update(collect_metric_field_names(result.strategy_savings))
        field_names.update(collect_metric_field_names(result.decisions))
        assert field_names.isdisjoint(_TOKEN_NAMED_METRIC_FIELDS)


def test_every_packing_case_uses_max_chars_and_chars_budget_unit() -> None:
    for case in _packing_cases():
        assert case.max_chars is not None
        assert case.max_chars > 0
        result = evaluate_case(case)
        packing_decisions = [
            entry
            for entry in result.decisions
            if entry.get("layer") == STRATEGY_BUDGET_AWARE_PACKING
        ]
        assert packing_decisions
        assert packing_decisions[0].get("budget_unit") == "chars"
        assert packing_decisions[0].get("max_chars") == case.max_chars


def test_every_protected_region_case_declares_protected_region_required() -> None:
    for case in STRONGER_OPTIMIZER_CORPUS:
        if case.expected.protected_region_required:
            assert case.case_id == "stronger_opt.packing_protected_region"
        if case.case_id == "stronger_opt.packing_protected_region":
            assert case.expected.protected_region_required is True


def test_mixed_cases_report_separated_strategy_attribution() -> None:
    mixed = _case_by_id("stronger_opt.mixed_dedupe_and_packing")
    result = evaluate_case(mixed)
    assert result.strategy_savings[STRATEGY_DEDUPLICATION] > 0
    assert result.strategy_savings[STRATEGY_BUDGET_AWARE_PACKING] > 0


# --- Evaluation behavior tests ---


def test_dedupe_case_produces_deduplication_char_savings() -> None:
    case = _case_by_id("stronger_opt.dedupe_rag_duplicate_lines")
    result = evaluate_case(case)

    assert result.total_saved_chars > 0
    assert result.strategy_savings[STRATEGY_DEDUPLICATION] > 0
    assert result.fallback_used is False


def test_packing_case_preserves_must_keep_and_drops_droppable_first() -> None:
    case = _case_by_id("stronger_opt.packing_priority_tiers")
    result = evaluate_case(case)

    assert result.total_saved_chars > 0
    assert result.strategy_savings[STRATEGY_BUDGET_AWARE_PACKING] > 0
    packing_entry = next(
        entry for entry in result.decisions if entry["layer"] == STRATEGY_BUDGET_AWARE_PACKING
    )
    decisions = {
        entry["fragment_id"]: entry for entry in packing_entry["packing_decisions"]  # type: ignore[index]
    }
    assert decisions["mk"]["decision"] == ContextPackingDecisionKind.KEEP.value
    assert decisions["high"]["decision"] == ContextPackingDecisionKind.KEEP.value
    assert decisions["drop"]["decision"] == ContextPackingDecisionKind.DROP.value
    assert decisions["comp"]["decision"] == ContextPackingDecisionKind.DROP.value


def test_compressible_case_reports_compaction_without_semantic_summarization() -> None:
    case = _case_by_id("stronger_opt.packing_compressible_whitespace")
    result = evaluate_case(case)

    assert result.strategy_savings[STRATEGY_BUDGET_AWARE_PACKING] > 0
    packing_entry = next(
        entry for entry in result.decisions if entry["layer"] == STRATEGY_BUDGET_AWARE_PACKING
    )
    decisions = {
        entry["fragment_id"]: entry for entry in packing_entry["packing_decisions"]  # type: ignore[index]
    }
    assert decisions["comp"]["decision"] == ContextPackingDecisionKind.COMPACT.value
    assert decisions["comp"]["reason"] == "compressible_whitespace_compacted"


def test_must_keep_over_budget_case_reports_fallback() -> None:
    case = _case_by_id("stronger_opt.packing_must_keep_over_budget")
    result = evaluate_case(case)

    assert result.fallback_used is True
    assert result.total_saved_chars == 0
    packing_entry = next(
        entry for entry in result.decisions if entry["layer"] == STRATEGY_BUDGET_AWARE_PACKING
    )
    assert packing_entry["decision"] == "fallback"


def test_noop_case_reports_no_savings_and_no_fallback() -> None:
    case = _case_by_id("stronger_opt.noop_clean_context")
    result = evaluate_case(case)

    assert result.total_saved_chars == 0
    assert result.fallback_used is False
    assert result.strategy_savings[STRATEGY_NO_OP] == 0


def test_mixed_case_reports_dedupe_and_packing_attribution_separately() -> None:
    case = _case_by_id("stronger_opt.mixed_dedupe_and_packing")
    result = evaluate_case(case)

    assert result.strategy_savings[STRATEGY_DEDUPLICATION] > 0
    assert result.strategy_savings[STRATEGY_BUDGET_AWARE_PACKING] > 0
    layers = {entry["layer"] for entry in result.decisions}
    assert STRATEGY_DEDUPLICATION in layers
    assert STRATEGY_BUDGET_AWARE_PACKING in layers


def test_report_contains_no_raw_content() -> None:
    for case in STRONGER_OPTIMIZER_CORPUS:
        result = evaluate_case(case)
        report = build_safe_evaluation_report(result, case)

        assert result.raw_content_in_report is False
        restricted_report = {
            key: value for key, value in report.items() if key not in {"safety_notes", "title"}
        }
        report_values = set()
        for value in _collect_string_values(restricted_report):
            report_values.add(value)
        if len(case.current_content) > 12:
            assert case.current_content not in report_values
        for fragment in case.fragments:
            if len(fragment.content) > 12 and fragment.content not in _PRIORITY_VALUES:
                assert fragment.content not in report_values


def test_droppable_optional_context_excluded_by_default() -> None:
    case = _case_by_id("stronger_opt.packing_droppable_excluded_default")
    result = evaluate_case(case)

    packing_entry = next(
        entry for entry in result.decisions if entry["layer"] == STRATEGY_BUDGET_AWARE_PACKING
    )
    decisions = {
        entry["fragment_id"]: entry for entry in packing_entry["packing_decisions"]  # type: ignore[index]
    }
    assert decisions["drop"]["decision"] == ContextPackingDecisionKind.DROP.value
    assert decisions["drop"]["reason"] == "droppable_excluded_by_default"


def test_protected_region_case_preserves_must_keep_marker() -> None:
    case = _case_by_id("stronger_opt.packing_protected_region")
    result = evaluate_case(case)

    assert result.fallback_used is False
    assert result.strategy_savings[STRATEGY_BUDGET_AWARE_PACKING] > 0
    packing_entry = next(
        entry for entry in result.decisions if entry["layer"] == STRATEGY_BUDGET_AWARE_PACKING
    )
    decisions = {
        entry["fragment_id"]: entry for entry in packing_entry["packing_decisions"]  # type: ignore[index]
    }
    assert decisions["mk"]["decision"] == ContextPackingDecisionKind.KEEP.value
    assert decisions["mk"]["output_chars"] > 0
