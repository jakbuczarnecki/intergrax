# © Artur Czarnecki. All rights reserved.

"""Safe regression benchmark report artifact builder (Phase TOKEN-OBS-2A)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from intergrax.runtime.token_optimization.emission import TokenOptimizationEmissionStatus
from intergrax.runtime.token_optimization.regression import (
    TokenRegressionResult,
    TokenRegressionSummary,
)
from intergrax.runtime.token_optimization.regression_emission import (
    TokenRegressionEmissionRunResult,
)
from intergrax.runtime.token_optimization.signals import sanitize_signal_metadata

_REPORT_ID_PREFIX = "token_regression_report_"
_REPORT_ID_HASH_LENGTH = 16

_UNSAFE_OUTPUT_KEYS: frozenset[str] = frozenset(
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


@dataclass(frozen=True, slots=True)
class TokenRegressionReportItem:
    """Safe scalar benchmark row for a regression report."""

    fixture_id: str
    source_type: str | None
    token_category: str | None
    strategy_id: str | None
    baseline_tokens: int
    optimized_tokens: int
    saved_tokens: int
    saved_ratio: float
    validation_status: str
    fallback_status: bool
    receipt_id: str | None
    passed: bool
    failure_reasons: tuple[str, ...]
    eval_case: str | None = None
    expected_behavior: str | None = None
    expectation_status: str | None = None


@dataclass(frozen=True, slots=True)
class TokenRegressionEmissionReport:
    """Aggregate emission disposition counts for a regression benchmark run."""

    attempted_result_emissions: int
    summary_emission_attempted: bool
    emitted_event_count: int
    emitted: int
    skipped_disabled: int
    skipped_kind_disabled: int
    dry_run: int


@dataclass(frozen=True, slots=True)
class TokenRegressionReport:
    """Redaction-safe regression benchmark report artifact."""

    report_id: str
    generated_at: str
    total_fixtures: int
    passed: int
    failed: int
    total_baseline_tokens: int
    total_optimized_tokens: int
    total_saved_tokens: int
    total_saved_ratio: float | None
    results: tuple[TokenRegressionReportItem, ...]
    emission: TokenRegressionEmissionReport | None
    metadata: Mapping[str, Any]


def build_token_regression_report(
    summary: TokenRegressionSummary,
    *,
    emission_run: TokenRegressionEmissionRunResult | None = None,
    report_id: str | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TokenRegressionReport:
    """Build a redaction-safe report from a regression benchmark summary."""
    combined_metadata = dict(summary.metadata)
    if metadata:
        combined_metadata.update(metadata)

    total_saved_ratio = (
        None
        if summary.total_baseline_tokens <= 0
        else summary.total_saved_ratio
    )

    return TokenRegressionReport(
        report_id=report_id or _derive_report_id(summary),
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        total_fixtures=summary.total_fixtures,
        passed=summary.passed,
        failed=summary.failed,
        total_baseline_tokens=summary.total_baseline_tokens,
        total_optimized_tokens=summary.total_optimized_tokens,
        total_saved_tokens=summary.total_saved_tokens,
        total_saved_ratio=total_saved_ratio,
        results=tuple(_build_report_item(result) for result in summary.results),
        emission=_build_emission_report(emission_run) if emission_run is not None else None,
        metadata=sanitize_signal_metadata(combined_metadata),
    )


def token_regression_report_to_dict(report: TokenRegressionReport) -> dict[str, Any]:
    """Serialize a regression report for JSON output."""
    payload: dict[str, Any] = {
        "report_id": report.report_id,
        "generated_at": report.generated_at,
        "total_fixtures": report.total_fixtures,
        "passed": report.passed,
        "failed": report.failed,
        "total_baseline_tokens": report.total_baseline_tokens,
        "total_optimized_tokens": report.total_optimized_tokens,
        "total_saved_tokens": report.total_saved_tokens,
        "total_saved_ratio": report.total_saved_ratio,
        "results": [_report_item_to_dict(item) for item in report.results],
        "metadata": dict(report.metadata),
    }
    if report.emission is not None:
        payload["emission"] = _emission_report_to_dict(report.emission)
    else:
        payload["emission"] = None
    return payload


def format_token_regression_report(report: TokenRegressionReport) -> str:
    """Human-readable regression benchmark report."""
    ratio_text = (
        "n/a"
        if report.total_saved_ratio is None
        else f"{report.total_saved_ratio:.4f}"
    )
    lines = [
        "Token regression benchmark report",
        f"report_id={report.report_id} generated_at={report.generated_at}",
        (
            f"fixtures={report.total_fixtures} passed={report.passed} "
            f"failed={report.failed}"
        ),
        (
            f"tokens baseline={report.total_baseline_tokens} "
            f"optimized={report.total_optimized_tokens} "
            f"saved={report.total_saved_tokens} "
            f"ratio={ratio_text}"
        ),
    ]
    for item in report.results:
        status = "PASS" if item.passed else "FAIL"
        source = item.source_type or "unknown"
        lines.append(
            f"  [{status}] {item.fixture_id} "
            f"({source}) "
            f"baseline={item.baseline_tokens} "
            f"optimized={item.optimized_tokens} "
            f"saved={item.saved_tokens} "
            f"ratio={item.saved_ratio:.4f}"
        )
        if item.failure_reasons:
            for reason in item.failure_reasons:
                lines.append(f"         - {reason}")
    if report.emission is not None:
        emission = report.emission
        lines.append(
            "emission "
            f"attempted_results={emission.attempted_result_emissions} "
            f"summary_attempted={emission.summary_emission_attempted} "
            f"emitted_events={emission.emitted_event_count} "
            f"emitted={emission.emitted} "
            f"skipped_disabled={emission.skipped_disabled} "
            f"skipped_kind_disabled={emission.skipped_kind_disabled} "
            f"dry_run={emission.dry_run}"
        )
    return "\n".join(lines)


def _derive_report_id(summary: TokenRegressionSummary) -> str:
    payload = "|".join(
        [
            str(summary.total_fixtures),
            str(summary.passed),
            str(summary.failed),
            str(summary.total_baseline_tokens),
            *(result.fixture_id for result in summary.results),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_REPORT_ID_HASH_LENGTH]
    return f"{_REPORT_ID_PREFIX}{digest}"


def _build_report_item(result: TokenRegressionResult) -> TokenRegressionReportItem:
    return TokenRegressionReportItem(
        fixture_id=result.fixture_id,
        source_type=result.source_type,
        token_category=_extract_token_category(result),
        strategy_id=result.strategy,
        baseline_tokens=result.baseline_tokens,
        optimized_tokens=result.optimized_tokens,
        saved_tokens=result.saved_tokens,
        saved_ratio=result.saved_ratio,
        validation_status=result.validation_status,
        fallback_status=result.fallback_status,
        receipt_id=_extract_receipt_id(result),
        passed=result.passed,
        failure_reasons=result.failure_reasons,
        eval_case=_extract_eval_metadata(result, "eval_case"),
        expected_behavior=_extract_eval_metadata(result, "expected_behavior"),
        expectation_status=_extract_eval_metadata(result, "expectation_status"),
    )


def _extract_token_category(result: TokenRegressionResult) -> str | None:
    for key in ("token_category", "category"):
        value = result.metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_receipt_id(result: TokenRegressionResult) -> str | None:
    receipt_id = result.metadata.get("receipt_id")
    if isinstance(receipt_id, str) and receipt_id:
        return receipt_id
    return None


def _extract_eval_metadata(result: TokenRegressionResult, key: str) -> str | None:
    value = result.metadata.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _build_emission_report(
    emission_run: TokenRegressionEmissionRunResult,
) -> TokenRegressionEmissionReport:
    emissions = list(emission_run.result_emissions)
    if emission_run.summary_emission is not None:
        emissions.append(emission_run.summary_emission)

    status_counts = {
        TokenOptimizationEmissionStatus.EMITTED.value: 0,
        TokenOptimizationEmissionStatus.SKIPPED_DISABLED.value: 0,
        TokenOptimizationEmissionStatus.SKIPPED_KIND_DISABLED.value: 0,
        TokenOptimizationEmissionStatus.DRY_RUN.value: 0,
    }
    for emission in emissions:
        status = (
            emission.status.value
            if isinstance(emission.status, TokenOptimizationEmissionStatus)
            else str(emission.status)
        )
        if status in status_counts:
            status_counts[status] += 1

    return TokenRegressionEmissionReport(
        attempted_result_emissions=len(emission_run.result_emissions),
        summary_emission_attempted=emission_run.summary_emission is not None,
        emitted_event_count=emission_run.emitted_event_count,
        emitted=status_counts[TokenOptimizationEmissionStatus.EMITTED.value],
        skipped_disabled=status_counts[TokenOptimizationEmissionStatus.SKIPPED_DISABLED.value],
        skipped_kind_disabled=status_counts[
            TokenOptimizationEmissionStatus.SKIPPED_KIND_DISABLED.value
        ],
        dry_run=status_counts[TokenOptimizationEmissionStatus.DRY_RUN.value],
    )


def _report_item_to_dict(item: TokenRegressionReportItem) -> dict[str, Any]:
    return {
        "fixture_id": item.fixture_id,
        "source_type": item.source_type,
        "token_category": item.token_category,
        "strategy_id": item.strategy_id,
        "baseline_tokens": item.baseline_tokens,
        "optimized_tokens": item.optimized_tokens,
        "saved_tokens": item.saved_tokens,
        "saved_ratio": item.saved_ratio,
        "validation_status": item.validation_status,
        "fallback_status": item.fallback_status,
        "receipt_id": item.receipt_id,
        "passed": item.passed,
        "failure_reasons": list(item.failure_reasons),
        "eval_case": item.eval_case,
        "expected_behavior": item.expected_behavior,
        "expectation_status": item.expectation_status,
    }


def _emission_report_to_dict(emission: TokenRegressionEmissionReport) -> dict[str, Any]:
    return {
        "attempted_result_emissions": emission.attempted_result_emissions,
        "summary_emission_attempted": emission.summary_emission_attempted,
        "emitted_event_count": emission.emitted_event_count,
        "emitted": emission.emitted,
        "skipped_disabled": emission.skipped_disabled,
        "skipped_kind_disabled": emission.skipped_kind_disabled,
        "dry_run": emission.dry_run,
    }
