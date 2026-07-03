# © Artur Czarnecki. All rights reserved.

"""TOKEN-6B: Token regression benchmark runner tests."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
    TokenOptimizationDecision,
    TokenOptimizationResult,
)
from intergrax.runtime.token_optimization.regression import (
    TokenRegressionBenchmarkRunner,
    TokenRegressionExpectation,
    TokenRegressionFixture,
    TokenRegressionSourceType,
    default_regression_fixtures,
    default_token_counter,
    run_token_regression_benchmark_execution,
    run_token_regression_benchmarks,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_token_regression_benchmarks.py"


def test_run_token_regression_benchmark_execution_returns_summary_plus_cases() -> None:
    execution = run_token_regression_benchmark_execution()

    assert len(execution.cases) == execution.summary.total_fixtures
    for case in execution.cases:
        assert case.result.fixture_id == case.fixture_id
        if case.runner_error is None:
            assert case.outcome is not None


def test_runner_measures_baseline_optimized_saved_and_ratio() -> None:
    summary = run_token_regression_benchmarks(token_counter=default_token_counter)
    result = summary.results[0]

    assert result.baseline_tokens > 0
    assert result.optimized_tokens >= 0
    assert result.saved_tokens == result.baseline_tokens - result.optimized_tokens
    if result.baseline_tokens > 0:
        assert result.saved_ratio == pytest.approx(
            result.saved_tokens / result.baseline_tokens
        )


def test_runner_supports_tool_schema_fixture() -> None:
    summary = run_token_regression_benchmarks()
    tool_results = [
        result
        for result in summary.results
        if result.source_type == TokenRegressionSourceType.TOOL_SCHEMA.value
    ]
    assert len(tool_results) == 2
    compact = next(r for r in tool_results if r.fixture_id == "tool_schema.compact_catalog")
    assert compact.passed is True
    assert compact.saved_tokens >= 1
    assert compact.metadata["eval_case"] == "compactable"


def test_runner_supports_context_pack_fixture() -> None:
    summary = run_token_regression_benchmarks()
    context_results = [
        result
        for result in summary.results
        if result.source_type == TokenRegressionSourceType.CONTEXT_PACK.value
    ]
    assert len(context_results) == 2
    compact = next(r for r in context_results if r.fixture_id == "context_pack.compact_fragments")
    assert compact.passed is True
    assert compact.saved_tokens >= 1
    assert compact.metadata["eval_case"] == "compactable"


def test_runner_supports_memory_summary_fixture() -> None:
    summary = run_token_regression_benchmarks()
    memory_results = [
        result
        for result in summary.results
        if result.source_type == TokenRegressionSourceType.MEMORY_SUMMARY.value
    ]
    assert len(memory_results) == 3
    result = next(r for r in memory_results if r.fixture_id == "memory_summary.compact_summary")
    assert result.passed is True
    assert result.receipt_present is True
    assert result.validation_status in {"passed", "not_applicable"}
    assert result.optimized_tokens <= result.baseline_tokens


def test_summary_aggregates_totals_correctly() -> None:
    summary = run_token_regression_benchmarks()

    assert summary.total_fixtures == len(summary.results)
    assert summary.passed + summary.failed == summary.total_fixtures
    assert summary.total_baseline_tokens == sum(
        result.baseline_tokens for result in summary.results
    )
    assert summary.total_optimized_tokens == sum(
        result.optimized_tokens for result in summary.results
    )
    assert summary.total_saved_tokens == (
        summary.total_baseline_tokens - summary.total_optimized_tokens
    )
    if summary.total_baseline_tokens > 0:
        assert summary.total_saved_ratio == pytest.approx(
            summary.total_saved_tokens / summary.total_baseline_tokens
        )


def test_fixture_fails_when_optimized_tokens_grow_unexpectedly() -> None:
    failing_fixture = TokenRegressionFixture(
        fixture_id="synthetic.token_growth",
        source_type=TokenRegressionSourceType.TOOL_SCHEMA,
        description="Synthetic fixture that reports token growth.",
        expectation=TokenRegressionExpectation(
            expected_min_saved_tokens=1,
            require_receipt=False,
            expect_validation_pass=False,
            allow_fallback=True,
        ),
        runner=lambda _counter: _SyntheticOutcome(
            original_content="one two three four",
            optimized_content="one two three four five six seven",
        ),
    )
    summary = run_token_regression_benchmarks(fixtures=[failing_fixture])

    assert summary.failed == 1
    assert summary.results[0].passed is False
    assert any("exceeded baseline_tokens" in reason for reason in summary.results[0].failure_reasons)


def test_fixture_fails_when_required_receipt_is_missing() -> None:
    failing_fixture = TokenRegressionFixture(
        fixture_id="synthetic.missing_receipt",
        source_type=TokenRegressionSourceType.CONTEXT_PACK,
        description="Synthetic fixture without receipt.",
        expectation=TokenRegressionExpectation(
            expected_min_saved_tokens=0,
            require_receipt=True,
            expect_validation_pass=False,
            allow_fallback=True,
        ),
        runner=lambda counter: _SyntheticOutcome(
            original_content="alpha beta gamma delta",
            optimized_content="alpha beta",
            token_counter=counter,
        ),
    )
    summary = run_token_regression_benchmarks(fixtures=[failing_fixture])

    assert summary.results[0].passed is False
    assert "required receipt missing" in summary.results[0].failure_reasons


def test_fixture_fails_when_validation_fails_unexpectedly() -> None:
    failing_fixture = TokenRegressionFixture(
        fixture_id="synthetic.validation_failed",
        source_type=TokenRegressionSourceType.MEMORY_SUMMARY,
        description="Synthetic fixture with failed validation.",
        expectation=TokenRegressionExpectation(
            expected_min_saved_tokens=0,
            require_receipt=False,
            expect_validation_pass=True,
            allow_fallback=True,
        ),
        runner=lambda counter: _SyntheticOutcome(
            original_content="alpha beta gamma",
            optimized_content="alpha beta",
            token_counter=counter,
            validation_status=ProtectedRegionValidationStatus.FAILED,
        ),
    )
    summary = run_token_regression_benchmarks(fixtures=[failing_fixture])

    assert summary.results[0].passed is False
    assert any(
        "validation status was not pass-like" in reason
        for reason in summary.results[0].failure_reasons
    )


def test_fixture_fails_when_validation_status_is_unknown_and_pass_expected() -> None:
    failing_fixture = TokenRegressionFixture(
        fixture_id="synthetic.validation_unknown",
        source_type=TokenRegressionSourceType.MEMORY_SUMMARY,
        description="Synthetic fixture with unknown validation status.",
        expectation=TokenRegressionExpectation(
            expected_min_saved_tokens=0,
            require_receipt=False,
            expect_validation_pass=True,
            allow_fallback=True,
        ),
        runner=lambda counter: _SyntheticOutcome(
            original_content="alpha beta gamma",
            optimized_content="alpha beta",
            token_counter=counter,
            validation_status="unknown",
        ),
    )
    summary = run_token_regression_benchmarks(fixtures=[failing_fixture])

    assert summary.results[0].passed is False
    assert summary.results[0].validation_status == "unknown"
    assert any(
        "validation status was not pass-like (status=unknown)" in reason
        for reason in summary.results[0].failure_reasons
    )


def test_fixture_fails_when_validation_metadata_is_missing_and_pass_expected() -> None:
    failing_fixture = TokenRegressionFixture(
        fixture_id="synthetic.validation_missing",
        source_type=TokenRegressionSourceType.CONTEXT_PACK,
        description="Synthetic fixture without validation metadata.",
        expectation=TokenRegressionExpectation(
            expected_min_saved_tokens=0,
            require_receipt=False,
            expect_validation_pass=True,
            allow_fallback=True,
        ),
        runner=lambda counter: _SyntheticOutcomeNoValidation(
            original_content="alpha beta gamma delta",
            optimized_content="alpha beta",
            token_counter=counter,
        ),
    )
    summary = run_token_regression_benchmarks(fixtures=[failing_fixture])

    assert summary.results[0].passed is False
    assert summary.results[0].validation_status == "missing"
    assert any(
        "validation status was not pass-like (status=missing)" in reason
        for reason in summary.results[0].failure_reasons
    )


def test_fixture_fails_when_validation_status_is_unexpected_and_pass_expected() -> None:
    failing_fixture = TokenRegressionFixture(
        fixture_id="synthetic.validation_unexpected",
        source_type=TokenRegressionSourceType.TOOL_SCHEMA,
        description="Synthetic fixture with unexpected validation status.",
        expectation=TokenRegressionExpectation(
            expected_min_saved_tokens=0,
            require_receipt=False,
            expect_validation_pass=True,
            allow_fallback=True,
        ),
        runner=lambda counter: _SyntheticOutcome(
            original_content="alpha beta gamma",
            optimized_content="alpha beta",
            token_counter=counter,
            validation_status="unexpected_status",
        ),
    )
    summary = run_token_regression_benchmarks(fixtures=[failing_fixture])

    assert summary.results[0].passed is False
    assert summary.results[0].validation_status == "unexpected_status"
    assert any(
        "validation status was not pass-like (status=unexpected_status)" in reason
        for reason in summary.results[0].failure_reasons
    )


def test_fixture_passes_when_validation_status_is_not_applicable_and_pass_expected() -> None:
    passing_fixture = TokenRegressionFixture(
        fixture_id="synthetic.validation_not_applicable",
        source_type=TokenRegressionSourceType.MEMORY_SUMMARY,
        description="Synthetic fixture with not_applicable validation.",
        expectation=TokenRegressionExpectation(
            expected_min_saved_tokens=0,
            require_receipt=False,
            expect_validation_pass=True,
            allow_fallback=True,
        ),
        runner=lambda counter: _SyntheticOutcome(
            original_content="alpha beta gamma",
            optimized_content="alpha beta",
            token_counter=counter,
            validation_status=ProtectedRegionValidationStatus.NOT_APPLICABLE,
        ),
    )
    summary = run_token_regression_benchmarks(fixtures=[passing_fixture])

    assert summary.results[0].passed is True
    assert summary.results[0].validation_status == ProtectedRegionValidationStatus.NOT_APPLICABLE.value


def test_fixture_does_not_fail_on_unknown_validation_when_pass_not_expected() -> None:
    passing_fixture = TokenRegressionFixture(
        fixture_id="synthetic.validation_unknown_allowed",
        source_type=TokenRegressionSourceType.TOOL_SCHEMA,
        description="Unknown validation allowed when pass not expected.",
        expectation=TokenRegressionExpectation(
            expected_min_saved_tokens=0,
            require_receipt=False,
            expect_validation_pass=False,
            allow_fallback=True,
        ),
        runner=lambda counter: _SyntheticOutcome(
            original_content="alpha beta gamma",
            optimized_content="alpha beta",
            token_counter=counter,
            validation_status="unknown",
        ),
    )
    summary = run_token_regression_benchmarks(fixtures=[passing_fixture])

    assert summary.results[0].passed is True
    assert summary.results[0].validation_status == "unknown"
    assert not any(
        "validation status was not pass-like" in reason
        for reason in summary.results[0].failure_reasons
    )


def test_fallback_allowed_or_rejected_by_fixture_expectation() -> None:
    allowed_fixture = TokenRegressionFixture(
        fixture_id="synthetic.fallback_allowed",
        source_type=TokenRegressionSourceType.TOOL_SCHEMA,
        description="Fallback allowed.",
        expectation=TokenRegressionExpectation(
            require_receipt=False,
            expect_validation_pass=False,
            allow_fallback=True,
        ),
        runner=lambda counter: _SyntheticOutcome(
            original_content="one two three",
            optimized_content="one two",
            token_counter=counter,
            fallback_status=True,
        ),
    )
    rejected_fixture = TokenRegressionFixture(
        fixture_id="synthetic.fallback_rejected",
        source_type=TokenRegressionSourceType.TOOL_SCHEMA,
        description="Fallback rejected.",
        expectation=TokenRegressionExpectation(
            require_receipt=False,
            expect_validation_pass=False,
            allow_fallback=False,
        ),
        runner=lambda counter: _SyntheticOutcome(
            original_content="one two three",
            optimized_content="one two",
            token_counter=counter,
            fallback_status=True,
        ),
    )

    allowed_summary = run_token_regression_benchmarks(fixtures=[allowed_fixture])
    rejected_summary = run_token_regression_benchmarks(fixtures=[rejected_fixture])

    assert allowed_summary.results[0].passed is True
    assert rejected_summary.results[0].passed is False
    assert "fallback used but allow_fallback=False" in rejected_summary.results[0].failure_reasons


def test_script_exits_zero_for_default_passing_fixtures() -> None:
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "fixtures=7 passed=7 failed=0" in completed.stdout


def test_script_exits_non_zero_for_failing_fixture() -> None:
    import importlib.util

    failing_fixture = TokenRegressionFixture(
        fixture_id="synthetic.script_failure",
        source_type=TokenRegressionSourceType.TOOL_SCHEMA,
        description="Force script failure path.",
        expectation=TokenRegressionExpectation(require_receipt=True),
        runner=lambda _counter: _SyntheticOutcome(
            original_content="one two three",
            optimized_content="one two",
        ),
    )
    spec = importlib.util.spec_from_file_location(
        "check_token_regression_benchmarks",
        _SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with patch.object(
        module,
        "run_token_regression_benchmarks",
        return_value=run_token_regression_benchmarks(fixtures=[failing_fixture]),
    ):
        exit_code = module.main([])

    assert exit_code != 0


def test_script_json_output_is_valid() -> None:
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--json"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["total_fixtures"] == 7
    assert payload["failed"] == 0
    assert len(payload["results"]) == 7


def test_runner_failure_path_surfaces_runner_exception() -> None:
    failing_fixture = TokenRegressionFixture(
        fixture_id="synthetic.runner_exception",
        source_type=TokenRegressionSourceType.TOOL_SCHEMA,
        description="Runner raises.",
        expectation=TokenRegressionExpectation(require_receipt=False),
        runner=lambda _counter: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    runner = TokenRegressionBenchmarkRunner(fixtures=[failing_fixture])
    summary = runner.run()

    assert summary.failed == 1
    assert summary.results[0].validation_status == "runner_error"
    assert any("runner_failed" in reason for reason in summary.results[0].failure_reasons)


def test_default_fixtures_include_all_source_categories() -> None:
    fixtures = default_regression_fixtures()
    source_types = {fixture.source_type for fixture in fixtures}
    assert source_types == {
        TokenRegressionSourceType.TOOL_SCHEMA,
        TokenRegressionSourceType.CONTEXT_PACK,
        TokenRegressionSourceType.MEMORY_SUMMARY,
    }
    assert len(fixtures) == 7


def test_default_eval_matrix_includes_required_categories() -> None:
    fixtures = default_regression_fixtures()
    eval_cases = {fixture.metadata["eval_case"] for fixture in fixtures}
    assert eval_cases == {"compactable", "protected", "fallback"}


def test_compactable_fixtures_save_tokens_above_configured_minimum() -> None:
    summary = run_token_regression_benchmarks()
    compactable = [
        result
        for result in summary.results
        if result.metadata.get("eval_case") == "compactable"
    ]
    assert len(compactable) == 3
    for result in compactable:
        fixture = next(
            item for item in default_regression_fixtures() if item.fixture_id == result.fixture_id
        )
        expectation = fixture.expectation
        if expectation.expected_min_saved_tokens is not None:
            assert result.saved_tokens >= expectation.expected_min_saved_tokens
        if (
            expectation.expected_min_saved_ratio is not None
            and expectation.expected_min_saved_ratio > 0
        ):
            assert result.saved_ratio >= expectation.expected_min_saved_ratio


def test_protected_fixtures_preserve_validation_and_block_savings() -> None:
    summary = run_token_regression_benchmarks()
    protected = [
        result
        for result in summary.results
        if result.metadata.get("eval_case") == "protected"
    ]
    assert len(protected) == 3
    for result in protected:
        assert result.passed is True
        assert result.saved_tokens == 0
        assert result.fallback_status is False
        assert result.validation_status in {"passed", "not_applicable"}
        assert result.metadata["expectation_status"] == "met"


def test_fallback_fixture_passes_by_falling_back() -> None:
    summary = run_token_regression_benchmarks()
    fallback = next(
        result
        for result in summary.results
        if result.fixture_id == "memory_summary.fallback_validation"
    )
    assert fallback.passed is True
    assert fallback.fallback_status is True
    assert fallback.saved_tokens == 0
    assert fallback.validation_status == "failed"
    assert fallback.metadata["expectation_status"] == "met"


class _SyntheticOutcome:
    """Minimal outcome stand-in for failing-path tests."""

    def __init__(
        self,
        *,
        original_content: str,
        optimized_content: str,
        token_counter: Callable[[str], int] | None = None,
        validation_status: ProtectedRegionValidationStatus | str = ProtectedRegionValidationStatus.PASSED,
        fallback_status: bool = False,
        receipt: object | None = None,
    ) -> None:
        self.original_content = original_content
        self.optimized_content = optimized_content
        if isinstance(validation_status, ProtectedRegionValidationStatus):
            self.protected_region_validation = ProtectedRegionValidationResult(
                status=validation_status,
            )
        self.receipt = receipt
        self.receipt_ref = None
        counter = token_counter or default_token_counter
        baseline = counter(original_content)
        optimized = counter(optimized_content)
        saved = baseline - optimized
        ratio = saved / baseline if baseline > 0 else 0.0
        self.original_tokens = baseline
        self.optimized_tokens = optimized
        self.saved_tokens = saved
        self.saved_ratio = ratio
        self.validation_status = (
            validation_status.value
            if isinstance(validation_status, ProtectedRegionValidationStatus)
            else validation_status
        )
        self.fallback_status = fallback_status
        self.result = TokenOptimizationResult(
            content=optimized_content,
            decision=TokenOptimizationDecision.APPLY,
            fallback_used=fallback_status,
        )


class _SyntheticOutcomeNoValidation:
    """Minimal outcome stand-in without validation metadata."""

    def __init__(
        self,
        *,
        original_content: str,
        optimized_content: str,
        token_counter: Callable[[str], int] | None = None,
        fallback_status: bool = False,
        receipt: object | None = None,
    ) -> None:
        self.original_content = original_content
        self.optimized_content = optimized_content
        self.receipt = receipt
        self.receipt_ref = None
        counter = token_counter or default_token_counter
        baseline = counter(original_content)
        optimized = counter(optimized_content)
        saved = baseline - optimized
        ratio = saved / baseline if baseline > 0 else 0.0
        self.original_tokens = baseline
        self.optimized_tokens = optimized
        self.saved_tokens = saved
        self.saved_ratio = ratio
        self.fallback_status = fallback_status
        self.result = TokenOptimizationResult(
            content=optimized_content,
            decision=TokenOptimizationDecision.APPLY,
            fallback_used=fallback_status,
        )
