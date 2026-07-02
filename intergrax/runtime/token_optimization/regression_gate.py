# © Artur Czarnecki. All rights reserved.

"""Formal regression benchmark gate over summary/report artifacts (Phase TOKEN-OBS-2C)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from intergrax.runtime.token_optimization.regression import (
    TokenRegressionResult,
    TokenRegressionSummary,
)
from intergrax.runtime.token_optimization.regression_report import TokenRegressionReport
from intergrax.runtime.token_optimization.signals import sanitize_signal_metadata

_REASON_MISSING_RECEIPT = "missing_receipt"
_REASON_FIXTURE_FAILED = "fixture_failed"
_REASON_EXPECTATION_NOT_MET = "expectation_not_met"
_REASON_UNEXPECTED_FALLBACK = "unexpected_fallback"
_REASON_TOTAL_SAVED_RATIO_BELOW = "total_saved_ratio_below_threshold"
_REASON_TOTAL_SAVED_TOKENS_BELOW = "total_saved_tokens_below_threshold"

_FAILURE_REASON_MISSING_RECEIPT = "required receipt missing"
_FAILURE_REASON_UNEXPECTED_FALLBACK = "unexpected fallback used"
_FAILURE_REASON_DISALLOWED_FALLBACK = "fallback used but allow_fallback=False"


class TokenRegressionGateStatus(StrEnum):
    """Regression gate disposition."""

    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class TokenRegressionGateFailure:
    """Single gate failure with a stable reason code."""

    fixture_id: str | None
    reason_code: str
    message: str


@dataclass(frozen=True, slots=True)
class TokenRegressionGateThresholds:
    """Configurable pass/fail thresholds for regression benchmark gates."""

    require_all_fixtures_passed: bool = True
    require_no_missing_receipts: bool = True
    require_no_unexpected_fallbacks: bool = True
    require_expectation_status_met: bool = True
    min_total_saved_ratio: float | None = None
    min_total_saved_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class TokenRegressionGateResult:
    """Outcome of evaluating regression benchmark gate thresholds."""

    status: TokenRegressionGateStatus
    total_fixtures: int
    passed: int
    failed: int
    total_saved_tokens: int
    total_saved_ratio: float
    failures: tuple[TokenRegressionGateFailure, ...]
    thresholds: TokenRegressionGateThresholds
    metadata: Mapping[str, Any]


def evaluate_token_regression_gate(
    summary: TokenRegressionSummary,
    *,
    thresholds: TokenRegressionGateThresholds | None = None,
    metadata: Mapping[str, Any] | None = None,
    report: TokenRegressionReport | None = None,
) -> TokenRegressionGateResult:
    """Evaluate formal gate thresholds against a regression benchmark summary."""
    active_thresholds = thresholds or TokenRegressionGateThresholds()
    combined_metadata = dict(summary.metadata)
    if metadata:
        combined_metadata.update(metadata)

    failures: list[TokenRegressionGateFailure] = []
    report_items_by_fixture = (
        {item.fixture_id: item for item in report.results} if report is not None else {}
    )

    for result in summary.results:
        report_item = report_items_by_fixture.get(result.fixture_id)
        failures.extend(
            _evaluate_fixture_gate(
                result,
                thresholds=active_thresholds,
                report_expectation_status=(
                    report_item.expectation_status if report_item is not None else None
                ),
                report_receipt_missing=(
                    report_item is not None and report_item.receipt_id is None
                ),
            )
        )

    if (
        active_thresholds.min_total_saved_ratio is not None
        and summary.total_saved_ratio < active_thresholds.min_total_saved_ratio
    ):
        failures.append(
            TokenRegressionGateFailure(
                fixture_id=None,
                reason_code=_REASON_TOTAL_SAVED_RATIO_BELOW,
                message=(
                    "total_saved_ratio "
                    f"({summary.total_saved_ratio:.4f}) below threshold "
                    f"({active_thresholds.min_total_saved_ratio:.4f})"
                ),
            )
        )

    if (
        active_thresholds.min_total_saved_tokens is not None
        and summary.total_saved_tokens < active_thresholds.min_total_saved_tokens
    ):
        failures.append(
            TokenRegressionGateFailure(
                fixture_id=None,
                reason_code=_REASON_TOTAL_SAVED_TOKENS_BELOW,
                message=(
                    "total_saved_tokens "
                    f"({summary.total_saved_tokens}) below threshold "
                    f"({active_thresholds.min_total_saved_tokens})"
                ),
            )
        )

    status = (
        TokenRegressionGateStatus.PASS
        if not failures
        else TokenRegressionGateStatus.FAIL
    )

    return TokenRegressionGateResult(
        status=status,
        total_fixtures=summary.total_fixtures,
        passed=summary.passed,
        failed=summary.failed,
        total_saved_tokens=summary.total_saved_tokens,
        total_saved_ratio=summary.total_saved_ratio,
        failures=tuple(failures),
        thresholds=active_thresholds,
        metadata=sanitize_signal_metadata(combined_metadata),
    )


def token_regression_gate_to_dict(result: TokenRegressionGateResult) -> dict[str, Any]:
    """Serialize a regression gate result for JSON output."""
    return {
        "status": result.status.value,
        "total_fixtures": result.total_fixtures,
        "passed": result.passed,
        "failed": result.failed,
        "total_saved_tokens": result.total_saved_tokens,
        "total_saved_ratio": result.total_saved_ratio,
        "failures": [
            {
                "fixture_id": failure.fixture_id,
                "reason_code": failure.reason_code,
                "message": failure.message,
            }
            for failure in result.failures
        ],
        "thresholds": _thresholds_to_dict(result.thresholds),
        "metadata": dict(result.metadata),
    }


def format_token_regression_gate(result: TokenRegressionGateResult) -> str:
    """Human-readable regression gate summary."""
    lines = [
        "Token regression gate",
        f"status={result.status.value}",
        (
            f"fixtures={result.total_fixtures} passed={result.passed} "
            f"failed={result.failed}"
        ),
        (
            f"tokens saved={result.total_saved_tokens} "
            f"ratio={result.total_saved_ratio:.4f}"
        ),
    ]
    for failure in result.failures:
        fixture = failure.fixture_id or "aggregate"
        lines.append(
            f"  [{failure.reason_code}] {fixture}: {failure.message}"
        )
    return "\n".join(lines)


def _evaluate_fixture_gate(
    result: TokenRegressionResult,
    *,
    thresholds: TokenRegressionGateThresholds,
    report_expectation_status: str | None,
    report_receipt_missing: bool,
) -> list[TokenRegressionGateFailure]:
    failures: list[TokenRegressionGateFailure] = []

    if thresholds.require_all_fixtures_passed and not result.passed:
        failures.append(
            TokenRegressionGateFailure(
                fixture_id=result.fixture_id,
                reason_code=_REASON_FIXTURE_FAILED,
                message=f"fixture {result.fixture_id} did not pass",
            )
        )

    if thresholds.require_no_missing_receipts and _is_missing_receipt(
        result,
        report_receipt_missing=report_receipt_missing,
    ):
        failures.append(
            TokenRegressionGateFailure(
                fixture_id=result.fixture_id,
                reason_code=_REASON_MISSING_RECEIPT,
                message=f"fixture {result.fixture_id} is missing a required receipt",
            )
        )

    if thresholds.require_expectation_status_met and _expectation_not_met(
        result,
        report_expectation_status=report_expectation_status,
    ):
        failures.append(
            TokenRegressionGateFailure(
                fixture_id=result.fixture_id,
                reason_code=_REASON_EXPECTATION_NOT_MET,
                message=(
                    f"fixture {result.fixture_id} expectation_status was not met"
                ),
            )
        )

    if thresholds.require_no_unexpected_fallbacks and _has_unexpected_fallback(
        result
    ):
        failures.append(
            TokenRegressionGateFailure(
                fixture_id=result.fixture_id,
                reason_code=_REASON_UNEXPECTED_FALLBACK,
                message=(
                    f"fixture {result.fixture_id} used an unexpected fallback"
                ),
            )
        )

    return failures


def _is_missing_receipt(
    result: TokenRegressionResult,
    *,
    report_receipt_missing: bool,
) -> bool:
    if result.receipt_present:
        return False
    if _failure_reason_indicates_missing_receipt(result.failure_reasons):
        return True
    return report_receipt_missing and not result.passed


def _failure_reason_indicates_missing_receipt(
    failure_reasons: tuple[str, ...],
) -> bool:
    return _FAILURE_REASON_MISSING_RECEIPT in failure_reasons


def _expectation_not_met(
    result: TokenRegressionResult,
    *,
    report_expectation_status: str | None,
) -> bool:
    status = result.metadata.get("expectation_status")
    if isinstance(status, str) and status:
        return status != "met"
    if report_expectation_status:
        return report_expectation_status != "met"
    return False


def _has_unexpected_fallback(result: TokenRegressionResult) -> bool:
    return any(
        reason == _FAILURE_REASON_UNEXPECTED_FALLBACK
        or reason == _FAILURE_REASON_DISALLOWED_FALLBACK
        for reason in result.failure_reasons
    )


def _thresholds_to_dict(
    thresholds: TokenRegressionGateThresholds,
) -> dict[str, Any]:
    return {
        "require_all_fixtures_passed": thresholds.require_all_fixtures_passed,
        "require_no_missing_receipts": thresholds.require_no_missing_receipts,
        "require_no_unexpected_fallbacks": thresholds.require_no_unexpected_fallbacks,
        "require_expectation_status_met": thresholds.require_expectation_status_met,
        "min_total_saved_ratio": thresholds.min_total_saved_ratio,
        "min_total_saved_tokens": thresholds.min_total_saved_tokens,
    }
