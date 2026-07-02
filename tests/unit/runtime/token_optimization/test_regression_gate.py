# © Artur Czarnecki. All rights reserved.

"""TOKEN-OBS-2C: regression benchmark gate threshold tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
    TokenOptimizationDecision,
    TokenOptimizationResult,
)
from intergrax.runtime.token_optimization.regression import (
    TokenRegressionExpectation,
    TokenRegressionFixture,
    TokenRegressionResult,
    TokenRegressionSourceType,
    TokenRegressionSummary,
    default_token_counter,
    run_token_regression_benchmarks,
)
from intergrax.runtime.token_optimization.regression_gate import (
    TokenRegressionGateResult,
    TokenRegressionGateStatus,
    TokenRegressionGateThresholds,
    evaluate_token_regression_gate,
    format_token_regression_gate,
    token_regression_gate_to_dict,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_REGRESSION_MODULE = (
    _REPO_ROOT / "intergrax" / "runtime" / "token_optimization" / "regression.py"
)
_REPORT_MODULE = (
    _REPO_ROOT / "intergrax" / "runtime" / "token_optimization" / "regression_report.py"
)
_EMISSION_MODULE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "token_optimization"
    / "regression_emission.py"
)
_BENCHMARK_SCRIPT = _REPO_ROOT / "scripts" / "check_token_regression_benchmarks.py"
_SCOPE_GUARD_MODULES = (
    _REGRESSION_MODULE,
    _REPORT_MODULE,
    _EMISSION_MODULE,
    _BENCHMARK_SCRIPT,
)
_SCOPE_GUARD_IMPORT = "regression_gate"

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
        "event",
        "signal",
    }
)


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


def _reason_codes(result: TokenRegressionGateResult) -> set[str]:
    return {failure.reason_code for failure in result.failures}


def test_default_benchmark_evaluates_to_pass() -> None:
    summary = run_token_regression_benchmarks()
    gate = evaluate_token_regression_gate(summary)

    assert gate.status == TokenRegressionGateStatus.PASS
    assert gate.total_fixtures == 7
    assert gate.passed == 7
    assert gate.failed == 0
    assert gate.failures == ()


def test_synthetic_failed_fixture_evaluates_to_fail_with_fixture_failed() -> None:
    summary = TokenRegressionSummary(
        total_fixtures=1,
        passed=0,
        failed=1,
        total_baseline_tokens=10,
        total_optimized_tokens=12,
        total_saved_tokens=-2,
        total_saved_ratio=-0.2,
        results=(
            TokenRegressionResult(
                fixture_id="synthetic.failed",
                source_type=TokenRegressionSourceType.TOOL_SCHEMA.value,
                strategy=None,
                baseline_tokens=10,
                optimized_tokens=12,
                saved_tokens=-2,
                saved_ratio=-0.2,
                validation_status="passed",
                fallback_status=False,
                receipt_present=True,
                passed=False,
                failure_reasons=(
                    "optimized_tokens (12) exceeded baseline_tokens (10)",
                ),
                metadata={"expectation_status": "failed"},
            ),
        ),
    )

    gate = evaluate_token_regression_gate(summary)

    assert gate.status == TokenRegressionGateStatus.FAIL
    assert "fixture_failed" in _reason_codes(gate)


def test_missing_receipt_evaluates_to_fail_with_missing_receipt() -> None:
    summary = run_token_regression_benchmarks(
        fixtures=[
            TokenRegressionFixture(
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
        ]
    )

    gate = evaluate_token_regression_gate(summary)

    assert gate.status == TokenRegressionGateStatus.FAIL
    assert "missing_receipt" in _reason_codes(gate)


def test_expectation_status_failed_evaluates_to_fail_with_expectation_not_met() -> None:
    summary = TokenRegressionSummary(
        total_fixtures=1,
        passed=0,
        failed=1,
        total_baseline_tokens=5,
        total_optimized_tokens=5,
        total_saved_tokens=0,
        total_saved_ratio=0.0,
        results=(
            TokenRegressionResult(
                fixture_id="synthetic.expectation_failed",
                source_type=TokenRegressionSourceType.MEMORY_SUMMARY.value,
                strategy=None,
                baseline_tokens=5,
                optimized_tokens=5,
                saved_tokens=0,
                saved_ratio=0.0,
                validation_status="passed",
                fallback_status=False,
                receipt_present=True,
                passed=False,
                failure_reasons=("saved_tokens (0) below expected_min_saved_tokens (1)",),
                metadata={"expectation_status": "failed"},
            ),
        ),
    )

    gate = evaluate_token_regression_gate(
        summary,
        thresholds=TokenRegressionGateThresholds(require_all_fixtures_passed=False),
    )

    assert gate.status == TokenRegressionGateStatus.FAIL
    assert "expectation_not_met" in _reason_codes(gate)


def test_unexpected_fallback_evaluates_to_fail_with_unexpected_fallback() -> None:
    summary = TokenRegressionSummary(
        total_fixtures=1,
        passed=0,
        failed=1,
        total_baseline_tokens=8,
        total_optimized_tokens=6,
        total_saved_tokens=2,
        total_saved_ratio=0.25,
        results=(
            TokenRegressionResult(
                fixture_id="synthetic.unexpected_fallback",
                source_type=TokenRegressionSourceType.MEMORY_SUMMARY.value,
                strategy=None,
                baseline_tokens=8,
                optimized_tokens=6,
                saved_tokens=2,
                saved_ratio=0.25,
                validation_status="passed",
                fallback_status=True,
                receipt_present=True,
                passed=False,
                failure_reasons=("unexpected fallback used",),
                metadata={"expectation_status": "failed"},
            ),
        ),
    )

    gate = evaluate_token_regression_gate(
        summary,
        thresholds=TokenRegressionGateThresholds(require_all_fixtures_passed=False),
    )

    assert gate.status == TokenRegressionGateStatus.FAIL
    assert "unexpected_fallback" in _reason_codes(gate)


def test_min_total_saved_ratio_threshold_can_fail() -> None:
    summary = run_token_regression_benchmarks()
    gate = evaluate_token_regression_gate(
        summary,
        thresholds=TokenRegressionGateThresholds(min_total_saved_ratio=1.0),
    )

    assert gate.status == TokenRegressionGateStatus.FAIL
    assert "total_saved_ratio_below_threshold" in _reason_codes(gate)


def test_min_total_saved_tokens_threshold_can_fail() -> None:
    summary = run_token_regression_benchmarks()
    gate = evaluate_token_regression_gate(
        summary,
        thresholds=TokenRegressionGateThresholds(
            min_total_saved_tokens=summary.total_saved_tokens + 1000
        ),
    )

    assert gate.status == TokenRegressionGateStatus.FAIL
    assert "total_saved_tokens_below_threshold" in _reason_codes(gate)


def test_dict_conversion_is_json_serializable() -> None:
    summary = run_token_regression_benchmarks()
    gate = evaluate_token_regression_gate(
        summary,
        metadata={"run_id": "gate-test-1"},
    )

    payload = token_regression_gate_to_dict(gate)
    serialized = json.dumps(payload, sort_keys=True)
    assert json.loads(serialized) == payload


def test_formatter_contains_status_and_failure_reason_codes() -> None:
    summary = TokenRegressionSummary(
        total_fixtures=1,
        passed=0,
        failed=1,
        total_baseline_tokens=4,
        total_optimized_tokens=4,
        total_saved_tokens=0,
        total_saved_ratio=0.0,
        results=(
            TokenRegressionResult(
                fixture_id="synthetic.format",
                source_type=TokenRegressionSourceType.TOOL_SCHEMA.value,
                strategy=None,
                baseline_tokens=4,
                optimized_tokens=4,
                saved_tokens=0,
                saved_ratio=0.0,
                validation_status="passed",
                fallback_status=False,
                receipt_present=False,
                passed=False,
                failure_reasons=("required receipt missing",),
                metadata={"expectation_status": "failed"},
            ),
        ),
    )
    gate = evaluate_token_regression_gate(summary)
    formatted = format_token_regression_gate(gate)

    assert "status=fail" in formatted
    assert "missing_receipt" in formatted
    assert "fixture_failed" in formatted


def test_unsafe_metadata_keys_are_sanitized() -> None:
    summary = run_token_regression_benchmarks()
    gate = evaluate_token_regression_gate(
        summary,
        metadata={
            "run_id": "gate-safe-1",
            "content": "secret prompt body",
            "original_content": "raw original",
            "payload": {"nested": "value"},
            "event": "raw event payload",
        },
    )

    payload = token_regression_gate_to_dict(gate)
    formatted = format_token_regression_gate(gate)

    assert _collect_keys(payload).isdisjoint(_UNSAFE_KEYS)
    assert "secret prompt body" not in formatted
    assert "raw original" not in formatted
    assert payload["metadata"]["run_id"] == "gate-safe-1"
    assert "content" not in payload["metadata"]
    assert "original_content" not in payload["metadata"]
    assert "payload" not in payload["metadata"]
    assert "event" not in payload["metadata"]


@pytest.mark.parametrize("module_path", _SCOPE_GUARD_MODULES, ids=lambda p: p.name)
def test_core_modules_and_benchmark_script_do_not_import_regression_gate(
    module_path: Path,
) -> None:
    source = module_path.read_text(encoding="utf-8")
    assert _SCOPE_GUARD_IMPORT not in source


class _SyntheticOutcome:
    """Minimal outcome stand-in for failing-path tests."""

    def __init__(
        self,
        *,
        original_content: str,
        optimized_content: str,
        token_counter: Callable[[str], int] | None = None,
        validation_status: ProtectedRegionValidationStatus | str = (
            ProtectedRegionValidationStatus.PASSED
        ),
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
