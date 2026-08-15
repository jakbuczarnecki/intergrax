# © Artur Czarnecki. All rights reserved.

"""TOKEN-8C: Pipeline configuration evaluation pack tests."""

from __future__ import annotations

import inspect
import json
import re

import pytest

from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationLayerDecision,
    TokenOptimizationPipelineMode,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
)
from intergrax.runtime.token_optimization.layers import BudgetAwarePackingInput
from tests.fixtures.token_optimization import pipeline_configuration_corpus as corpus

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_CASE_ID_PATTERN = re.compile(r"^pipeline_eval\.[a-z0-9_]+$")

_REQUIRED_CASE_IDS: tuple[str, ...] = (
    "pipeline_eval.rag_duplicate_lines",
    "pipeline_eval.rag_priority_packing",
    "pipeline_eval.rag_mixed_dedupe_packing",
    "pipeline_eval.tool_noisy_repeated_output",
    "pipeline_eval.tool_protected_value",
    "pipeline_eval.clean_noop",
)

_REQUIRED_CONFIGURATION_IDS: tuple[str, ...] = (
    "disabled",
    "measure_only",
    "exact_only",
    "extractive_allowed",
    "extractive_blocked",
    "packing_only",
    "exact_then_packing",
    "exact_then_extractive",
    "extractive_then_exact",
)

_BUILTIN_LAYER_IDS = frozenset(
    {
        "builtin.exact_deduplication",
        "builtin.extractive_filtering",
        "builtin.budget_aware_context_packing",
    }
)

_FORBIDDEN_REPORT_FIELD_NAMES = frozenset(
    {
        "content",
        "original_content",
        "final_content",
        "current_content",
        "output_content",
        "fragment_content",
        "request_metadata",
        "result_metadata",
        "receipt_metadata",
        "exception",
        "traceback",
    }
)

_RECOMMENDATION_SUBSTRINGS = (
    "winner",
    "best_configuration",
    "recommended_configuration",
    "recommendation",
    "production_ready",
    "quality_score",
)


def _result(case_id: str, configuration_id: str) -> corpus.PipelineConfigurationEvaluationResult:
    return next(
        result
        for result in corpus.run_pipeline_configuration_evaluation_matrix().results
        if result.case_id == case_id and result.configuration_id == configuration_id
    )


def _results_for_configuration(
    configuration_id: str,
) -> tuple[corpus.PipelineConfigurationEvaluationResult, ...]:
    return tuple(
        result
        for result in corpus.run_pipeline_configuration_evaluation_matrix().results
        if result.configuration_id == configuration_id
    )


# --- Test group A: corpus integrity ---


def test_corpus_contains_exactly_six_canonical_cases() -> None:
    assert len(corpus.PIPELINE_CONFIGURATION_CORPUS) == 6


def test_case_ids_are_exact_canonical_set() -> None:
    case_ids = [case.case_id for case in corpus.PIPELINE_CONFIGURATION_CORPUS]
    assert tuple(case_ids) == _REQUIRED_CASE_IDS


def test_case_ids_are_unique_and_match_stable_pattern() -> None:
    case_ids = [case.case_id for case in corpus.PIPELINE_CONFIGURATION_CORPUS]
    assert len(case_ids) == len(set(case_ids))
    for case_id in case_ids:
        assert _CASE_ID_PATTERN.match(case_id)


def test_every_case_uses_synthetic_marker() -> None:
    for case in corpus.PIPELINE_CONFIGURATION_CORPUS:
        assert case.synthetic_marker == corpus.PIPELINE_CONFIGURATION_SYNTHETIC_CORPUS_MARKER


def test_every_case_source_type_is_known() -> None:
    for case in corpus.PIPELINE_CONFIGURATION_CORPUS:
        assert case.source_type is not TokenOptimizationSourceType.UNKNOWN


def test_no_forbidden_secret_patterns_in_corpus() -> None:
    assert corpus.corpus_contains_forbidden_secret_patterns(corpus.PIPELINE_CONFIGURATION_CORPUS) == []


def test_only_packing_cases_contain_budget_aware_packing_input() -> None:
    packing_case_ids = {
        "pipeline_eval.rag_priority_packing",
        "pipeline_eval.rag_mixed_dedupe_packing",
    }
    for case in corpus.PIPELINE_CONFIGURATION_CORPUS:
        has_packing = any(
            isinstance(value, BudgetAwarePackingInput)
            for value in case.metadata.values()
        )
        if case.case_id in packing_case_ids:
            assert has_packing
        else:
            assert not has_packing


def test_protected_case_has_explicit_synthetic_protected_region() -> None:
    protected_case = next(
        case
        for case in corpus.PIPELINE_CONFIGURATION_CORPUS
        if case.case_id == "pipeline_eval.tool_protected_value"
    )
    assert len(protected_case.protected_regions) >= 1
    assert corpus.protected_synthetic_value().startswith("PROTECTED-SYNTH-")


# --- Test group B: configuration integrity ---


def test_matrix_contains_exactly_nine_configurations() -> None:
    assert len(corpus.PIPELINE_CONFIGURATION_MATRIX) == 9


def test_configuration_ids_are_exact_canonical_order() -> None:
    config_ids = [
        configuration.configuration_id
        for configuration in corpus.PIPELINE_CONFIGURATION_MATRIX
    ]
    assert tuple(config_ids) == _REQUIRED_CONFIGURATION_IDS


def test_configuration_ids_are_unique() -> None:
    config_ids = [
        configuration.configuration_id
        for configuration in corpus.PIPELINE_CONFIGURATION_MATRIX
    ]
    assert len(config_ids) == len(set(config_ids))


def test_selection_order_matches_layer_ref_order() -> None:
    for configuration in corpus.PIPELINE_CONFIGURATION_MATRIX:
        selection_ids = [selection.layer_id for selection in configuration.selections]
        ref_ids = [layer_ref.layer_id for layer_ref in configuration.layer_refs]
        assert selection_ids == ref_ids


def test_all_pipeline_modes_are_replace() -> None:
    for configuration in corpus.PIPELINE_CONFIGURATION_MATRIX:
        assert configuration.pipeline_mode is TokenOptimizationPipelineMode.REPLACE


def test_budget_aware_selections_use_typed_config() -> None:
    for configuration in corpus.PIPELINE_CONFIGURATION_MATRIX:
        for selection in configuration.selections:
            if selection.layer_id == "builtin.budget_aware_context_packing":
                assert selection.config is not None
                assert selection.config.max_chars > 0


def test_extractive_configurations_use_typed_config() -> None:
    for configuration in corpus.PIPELINE_CONFIGURATION_MATRIX:
        for selection in configuration.selections:
            if selection.layer_id == "builtin.extractive_filtering":
                assert selection.config is not None


def test_extractive_allowed_policy_allows_lossy() -> None:
    configuration = next(
        item
        for item in corpus.PIPELINE_CONFIGURATION_MATRIX
        if item.configuration_id == "extractive_allowed"
    )
    assert configuration.policy.allow_lossy is True


def test_extractive_blocked_policy_disallows_lossy() -> None:
    configuration = next(
        item
        for item in corpus.PIPELINE_CONFIGURATION_MATRIX
        if item.configuration_id == "extractive_blocked"
    )
    assert configuration.policy.allow_lossy is False


def test_disabled_policy_is_not_enabled() -> None:
    configuration = next(
        item
        for item in corpus.PIPELINE_CONFIGURATION_MATRIX
        if item.configuration_id == "disabled"
    )
    assert configuration.policy.enabled is False


def test_measure_only_policy_profile() -> None:
    configuration = next(
        item
        for item in corpus.PIPELINE_CONFIGURATION_MATRIX
        if item.configuration_id == "measure_only"
    )
    assert configuration.policy.profile is TokenOptimizationProfile.MEASURE_ONLY


def test_no_plugin_or_unknown_layer_ids() -> None:
    for configuration in corpus.PIPELINE_CONFIGURATION_MATRIX:
        for selection in configuration.selections:
            assert selection.layer_id in _BUILTIN_LAYER_IDS
        for layer_ref in configuration.layer_refs:
            assert layer_ref.layer_id in _BUILTIN_LAYER_IDS
            assert layer_ref.plugin_id is None


# --- Test group C: complete matrix ---


def test_complete_matrix_counts() -> None:
    execution = corpus.run_pipeline_configuration_evaluation_matrix()
    assert execution.case_count == 6
    assert execution.configuration_count == 9
    assert execution.execution_count == 54
    assert len(execution.results) == 54


def test_every_case_configuration_pair_appears_once() -> None:
    execution = corpus.run_pipeline_configuration_evaluation_matrix()
    pairs = [(result.case_id, result.configuration_id) for result in execution.results]
    assert len(pairs) == len(set(pairs))


def test_canonical_result_ordering() -> None:
    execution = corpus.run_pipeline_configuration_evaluation_matrix()
    expected_pairs: list[tuple[str, str]] = []
    for case in corpus.PIPELINE_CONFIGURATION_CORPUS:
        for configuration in corpus.PIPELINE_CONFIGURATION_MATRIX:
            expected_pairs.append((case.case_id, configuration.configuration_id))
    actual_pairs = [
        (result.case_id, result.configuration_id) for result in execution.results
    ]
    assert actual_pairs == expected_pairs


# --- Test group D: disabled behavior ---


def test_disabled_configuration_preserves_content_without_execution() -> None:
    for result in _results_for_configuration("disabled"):
        assert result.original_chars == result.final_chars
        assert result.char_delta == 0
        assert result.applied_layer_ids == ()
        assert result.executed_layer_ids == ()
        assert result.failed_layer_ids == ()
        assert result.fallback_used is False


# --- Test group E: measure-only behavior ---


def test_measure_only_configuration_executes_without_replacing_content() -> None:
    for result in _results_for_configuration("measure_only"):
        assert result.original_chars == result.final_chars
        assert result.char_delta == 0
        assert result.executed_layer_ids
        assert result.failed_layer_ids == ()
        assert result.fallback_used is False


# --- Test group F: exact deduplication proof ---


def test_exact_only_reduces_rag_duplicate_lines() -> None:
    result = _result("pipeline_eval.rag_duplicate_lines", "exact_only")
    assert result.char_delta > 0
    assert result.applied_layer_ids == ("builtin.exact_deduplication",)
    assert result.failed_layer_ids == ()
    assert result.fallback_used is False
    assert result.layer_outcomes[0].decision == TokenOptimizationLayerDecision.APPLY.value


# --- Test group G: extractive policy proof ---


def test_extractive_allowed_reduces_noisy_tool_output() -> None:
    result = _result("pipeline_eval.tool_noisy_repeated_output", "extractive_allowed")
    assert result.char_delta > 0
    assert "builtin.extractive_filtering" in result.applied_layer_ids
    assert result.failed_layer_ids == ()


def test_extractive_blocked_bypasses_with_policy_disallowed() -> None:
    result = _result("pipeline_eval.tool_noisy_repeated_output", "extractive_blocked")
    assert result.char_delta == 0
    assert result.applied_layer_ids == ()
    assert result.bypassed_layer_ids == ("builtin.extractive_filtering",)
    assert result.executed_layer_ids == ()
    assert result.layer_outcomes[0].bypass_reason == "policy_disallowed"


# --- Test group H: packing proof ---


def test_packing_only_reduces_priority_packing_case() -> None:
    result = _result("pipeline_eval.rag_priority_packing", "packing_only")
    assert result.char_delta > 0
    assert result.applied_layer_ids == ("builtin.budget_aware_context_packing",)
    assert result.executed_layer_ids == ("builtin.budget_aware_context_packing",)
    assert result.failed_layer_ids == ()


# --- Test group I: sequential RAG pipeline proof ---


def test_exact_then_packing_executes_both_layers_in_order() -> None:
    result = _result("pipeline_eval.rag_mixed_dedupe_packing", "exact_then_packing")
    assert result.executed_layer_ids == (
        "builtin.exact_deduplication",
        "builtin.budget_aware_context_packing",
    )
    assert result.applied_layer_ids == (
        "builtin.exact_deduplication",
        "builtin.budget_aware_context_packing",
    )
    assert result.char_delta > 0
    assert result.failed_layer_ids == ()
    assert result.fallback_used is False


# --- Test group J: layer-order proof ---


def test_noisy_tool_output_preserves_layer_order_per_configuration() -> None:
    exact_then = _result(
        "pipeline_eval.tool_noisy_repeated_output",
        "exact_then_extractive",
    )
    extractive_then = _result(
        "pipeline_eval.tool_noisy_repeated_output",
        "extractive_then_exact",
    )
    assert exact_then.executed_layer_ids == (
        "builtin.exact_deduplication",
        "builtin.extractive_filtering",
    )
    assert extractive_then.executed_layer_ids == (
        "builtin.extractive_filtering",
        "builtin.exact_deduplication",
    )
    assert exact_then.failed_layer_ids == ()
    assert extractive_then.failed_layer_ids == ()
    assert exact_then.char_delta > 0 or extractive_then.char_delta > 0


# --- Test group K: protected-region fallback ---


def test_protected_tool_output_falls_back_safely_under_extractive_allowed() -> None:
    result = _result("pipeline_eval.tool_protected_value", "extractive_allowed")
    assert result.fallback_used is True
    assert result.final_chars == result.original_chars
    assert result.char_delta == 0
    assert result.failed_layer_ids == ()
    extractive_outcome = result.layer_outcomes[0]
    assert extractive_outcome.decision == TokenOptimizationLayerDecision.FALLBACK.value
    assert extractive_outcome.validation_status == "failed"


# --- Test group L: clean no-op behavior ---


def test_clean_noop_never_reports_artificial_positive_reduction() -> None:
    execution = corpus.run_pipeline_configuration_evaluation_matrix()
    for result in execution.results:
        if result.case_id != "pipeline_eval.clean_noop":
            continue
        assert result.char_delta <= 0
        assert result.failed_layer_ids == ()


# --- Test group M: unsupported source behavior ---


def test_rag_duplicate_case_bypasses_extractive_as_unsupported_source() -> None:
    result = _result("pipeline_eval.rag_duplicate_lines", "extractive_allowed")
    assert result.char_delta == 0
    assert result.failed_layer_ids == ()
    assert result.bypassed_layer_ids == ("builtin.extractive_filtering",)
    assert result.layer_outcomes[0].bypass_reason == "unsupported_source_type"


def test_tool_output_bypasses_packing_as_unsupported_source() -> None:
    result = _result("pipeline_eval.tool_noisy_repeated_output", "packing_only")
    assert result.char_delta == 0
    assert result.failed_layer_ids == ()
    assert result.bypassed_layer_ids == ("builtin.budget_aware_context_packing",)
    assert result.layer_outcomes[0].bypass_reason == "unsupported_source_type"


# --- Test group N: determinism ---


def test_matrix_execution_is_deterministic() -> None:
    first = corpus.run_pipeline_configuration_evaluation_matrix()
    second = corpus.run_pipeline_configuration_evaluation_matrix()
    assert first == second
    assert corpus.build_safe_pipeline_configuration_report(
        first,
    ) == corpus.build_safe_pipeline_configuration_report(second)


# --- Test group O: raw-content-safe report ---


def test_report_excludes_raw_content_and_forbidden_fields() -> None:
    execution = corpus.run_pipeline_configuration_evaluation_matrix()
    report = corpus.build_safe_pipeline_configuration_report(execution)
    field_names = corpus.collect_report_field_names(report)
    assert field_names.isdisjoint(_FORBIDDEN_REPORT_FIELD_NAMES)

    serialized = json.dumps(report)
    for case in corpus.PIPELINE_CONFIGURATION_CORPUS:
        if len(case.content) > 12:
            assert case.content not in serialized
    assert corpus.protected_synthetic_value() not in serialized


# --- Test group P: character-only reporting ---


def test_report_uses_character_metrics_only() -> None:
    execution = corpus.run_pipeline_configuration_evaluation_matrix()
    report = corpus.build_safe_pipeline_configuration_report(execution)
    assert report["budget_unit"] == "chars"
    for result in execution.results:
        assert result.budget_unit == "chars"
        assert isinstance(result.original_chars, int)
        assert isinstance(result.final_chars, int)
        assert isinstance(result.char_delta, int)
        assert isinstance(result.reduction_ratio, float)

    for field_name in corpus.collect_report_field_names(report):
        assert "token" not in field_name.casefold()


# --- Test group Q: no recommendation semantics ---


def test_report_has_no_recommendation_semantics() -> None:
    execution = corpus.run_pipeline_configuration_evaluation_matrix()
    report = corpus.build_safe_pipeline_configuration_report(execution)
    field_names = {name.casefold() for name in corpus.collect_report_field_names(report)}
    for substring in _RECOMMENDATION_SUBSTRINGS:
        assert not any(substring in field_name for field_name in field_names)

    for value in corpus.collect_report_string_values(report):
        lowered = value.casefold()
        for substring in _RECOMMENDATION_SUBSTRINGS:
            assert substring not in lowered


# --- Test group R: standard execution path proof ---


def test_evaluator_uses_standard_token_optimization_pipeline_path() -> None:
    source = inspect.getsource(corpus.evaluate_pipeline_configuration)
    assert "create_builtin_token_optimization_layer_catalog" in source
    assert "TokenOptimizationPipelineRunner" in source
    assert "TokenOptimizationPipelineConfig" in source
    assert "TokenOptimizationPipelineMode.REPLACE" in source
    assert ".optimize(" not in source
