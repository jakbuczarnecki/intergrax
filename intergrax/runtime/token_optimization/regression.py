# © Artur Czarnecki. All rights reserved.

"""Deterministic token regression benchmark runner (Phase TOKEN-6B)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from intergrax.memory.summary_compressor import (
    DEFAULT_MEMORY_SUMMARY_TOKEN_POLICY,
    MemorySummaryCompressionOutcome,
    optimize_memory_summary,
)
from intergrax.runtime.token_optimization.context_pack import (
    DEFAULT_CONTEXT_PACK_TOKEN_POLICY,
    ContextFragment,
    ContextPackOptimizationOutcome,
    optimize_context_pack,
)
from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegionValidationStatus,
    TokenOptimizationStrategyRef,
)
from intergrax.runtime.token_optimization.tool_schema import (
    DEFAULT_TOOL_CATALOG_TOKEN_POLICY,
    ToolSchemaOptimizationOutcome,
    optimize_tool_schema_catalog,
)

TokenCounter = Callable[[str], int]
RegressionRunner = Callable[[TokenCounter], object]


class TokenRegressionSourceType(StrEnum):
    """Benchmark fixture source category."""

    TOOL_SCHEMA = "tool_schema"
    CONTEXT_PACK = "context_pack"
    MEMORY_SUMMARY = "memory_summary"


def default_token_counter(value: str) -> int:
    """Deterministic word-count token estimator (no external tokenizer)."""
    stripped = value.strip()
    if not stripped:
        return 0
    return len(stripped.split())


@dataclass(frozen=True, slots=True)
class TokenRegressionExpectation:
    """Pass/fail expectations for a single benchmark fixture."""

    expected_min_saved_tokens: int | None = None
    expected_min_saved_ratio: float | None = None
    require_receipt: bool = True
    expect_validation_pass: bool = True
    allow_fallback: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenRegressionFixture:
    """Deterministic benchmark fixture definition."""

    fixture_id: str
    source_type: TokenRegressionSourceType
    description: str
    expectation: TokenRegressionExpectation
    runner: RegressionRunner
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenRegressionResult:
    """Benchmark result for a single fixture."""

    fixture_id: str
    source_type: str
    strategy: str | None
    baseline_tokens: int
    optimized_tokens: int
    saved_tokens: int
    saved_ratio: float
    validation_status: str
    fallback_status: bool
    receipt_present: bool
    passed: bool
    failure_reasons: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenRegressionSummary:
    """Aggregated benchmark summary."""

    total_fixtures: int
    passed: int
    failed: int
    total_baseline_tokens: int
    total_optimized_tokens: int
    total_saved_tokens: int
    total_saved_ratio: float
    results: tuple[TokenRegressionResult, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TokenRegressionBenchmarkRunner:
    """Execute deterministic token optimization regression fixtures."""

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        fixtures: Sequence[TokenRegressionFixture] | None = None,
    ) -> None:
        self._token_counter = token_counter or default_token_counter
        self._fixtures = tuple(fixtures) if fixtures is not None else default_regression_fixtures()

    @property
    def token_counter(self) -> TokenCounter:
        return self._token_counter

    @property
    def fixtures(self) -> tuple[TokenRegressionFixture, ...]:
        return self._fixtures

    def run(self) -> TokenRegressionSummary:
        results = tuple(self._run_fixture(fixture) for fixture in self._fixtures)
        return _build_summary(results, metadata={"token_counter": "default_word_count"})

    def _run_fixture(self, fixture: TokenRegressionFixture) -> TokenRegressionResult:
        try:
            outcome = fixture.runner(self._token_counter)
        except Exception as exc:  # noqa: BLE001 — benchmark gate must surface runner failures
            return TokenRegressionResult(
                fixture_id=fixture.fixture_id,
                source_type=fixture.source_type.value,
                strategy=None,
                baseline_tokens=0,
                optimized_tokens=0,
                saved_tokens=0,
                saved_ratio=0.0,
                validation_status="runner_error",
                fallback_status=False,
                receipt_present=False,
                passed=False,
                failure_reasons=(f"runner_failed: {exc}",),
                metadata=dict(fixture.metadata),
            )

        baseline_tokens, optimized_tokens, saved_tokens, saved_ratio = _resolve_token_metrics(
            outcome,
            self._token_counter,
        )
        validation_status = _resolve_validation_status(outcome)
        fallback_status = _resolve_fallback_status(outcome)
        receipt_present = _resolve_receipt_present(outcome)
        strategy = _resolve_strategy_id(outcome)

        failure_reasons = _evaluate_expectations(
            expectation=fixture.expectation,
            baseline_tokens=baseline_tokens,
            optimized_tokens=optimized_tokens,
            saved_tokens=saved_tokens,
            saved_ratio=saved_ratio,
            validation_status=validation_status,
            fallback_status=fallback_status,
            receipt_present=receipt_present,
        )

        combined_metadata: dict[str, Any] = dict(fixture.metadata)
        combined_metadata.update(fixture.expectation.metadata)
        combined_metadata["description"] = fixture.description

        return TokenRegressionResult(
            fixture_id=fixture.fixture_id,
            source_type=fixture.source_type.value,
            strategy=strategy,
            baseline_tokens=baseline_tokens,
            optimized_tokens=optimized_tokens,
            saved_tokens=saved_tokens,
            saved_ratio=saved_ratio,
            validation_status=validation_status,
            fallback_status=fallback_status,
            receipt_present=receipt_present,
            passed=not failure_reasons,
            failure_reasons=failure_reasons,
            metadata=combined_metadata,
        )


def run_token_regression_benchmarks(
    *,
    token_counter: TokenCounter | None = None,
    fixtures: Sequence[TokenRegressionFixture] | None = None,
) -> TokenRegressionSummary:
    """Run the default or custom deterministic regression benchmark suite."""
    runner = TokenRegressionBenchmarkRunner(token_counter=token_counter, fixtures=fixtures)
    return runner.run()


def default_regression_fixtures() -> tuple[TokenRegressionFixture, ...]:
    """Built-in representative fixtures for helper-only optimizers."""
    return (
        TokenRegressionFixture(
            fixture_id="tool_schema.compact_catalog",
            source_type=TokenRegressionSourceType.TOOL_SCHEMA,
            description="Compact pretty-printed tool catalog JSON with whitespace savings.",
            expectation=TokenRegressionExpectation(
                expected_min_saved_tokens=1,
                require_receipt=True,
                expect_validation_pass=True,
                allow_fallback=False,
            ),
            runner=_run_tool_schema_fixture,
            metadata={"category": "tool_catalog"},
        ),
        TokenRegressionFixture(
            fixture_id="context_pack.compact_fragments",
            source_type=TokenRegressionSourceType.CONTEXT_PACK,
            description="Compact context pack fragments with whitespace normalization.",
            expectation=TokenRegressionExpectation(
                expected_min_saved_tokens=1,
                require_receipt=True,
                expect_validation_pass=True,
                allow_fallback=False,
            ),
            runner=_run_context_pack_fixture,
            metadata={"category": "rag_context_pack"},
        ),
        TokenRegressionFixture(
            fixture_id="memory_summary.compact_summary",
            source_type=TokenRegressionSourceType.MEMORY_SUMMARY,
            description="Compact memory summary with structural whitespace normalization.",
            expectation=TokenRegressionExpectation(
                require_receipt=True,
                expect_validation_pass=True,
                allow_fallback=False,
            ),
            runner=_run_memory_summary_fixture,
            metadata={"category": "memory"},
        ),
    )


def regression_result_to_dict(result: TokenRegressionResult) -> dict[str, Any]:
    """Serialize a single benchmark result for JSON output."""
    return {
        "fixture_id": result.fixture_id,
        "source_type": result.source_type,
        "strategy": result.strategy,
        "baseline_tokens": result.baseline_tokens,
        "optimized_tokens": result.optimized_tokens,
        "saved_tokens": result.saved_tokens,
        "saved_ratio": result.saved_ratio,
        "validation_status": result.validation_status,
        "fallback_status": result.fallback_status,
        "receipt_present": result.receipt_present,
        "passed": result.passed,
        "failure_reasons": list(result.failure_reasons),
        "metadata": dict(result.metadata),
    }


def regression_summary_to_dict(summary: TokenRegressionSummary) -> dict[str, Any]:
    """Serialize an aggregated benchmark summary for JSON output."""
    return {
        "total_fixtures": summary.total_fixtures,
        "passed": summary.passed,
        "failed": summary.failed,
        "total_baseline_tokens": summary.total_baseline_tokens,
        "total_optimized_tokens": summary.total_optimized_tokens,
        "total_saved_tokens": summary.total_saved_tokens,
        "total_saved_ratio": summary.total_saved_ratio,
        "results": [regression_result_to_dict(result) for result in summary.results],
        "metadata": dict(summary.metadata),
    }


def format_regression_summary(summary: TokenRegressionSummary) -> str:
    """Human-readable benchmark summary."""
    lines = [
        "Token regression benchmarks",
        (
            f"fixtures={summary.total_fixtures} passed={summary.passed} "
            f"failed={summary.failed}"
        ),
        (
            f"tokens baseline={summary.total_baseline_tokens} "
            f"optimized={summary.total_optimized_tokens} "
            f"saved={summary.total_saved_tokens} "
            f"ratio={summary.total_saved_ratio:.4f}"
        ),
    ]
    for result in summary.results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"  [{status}] {result.fixture_id} "
            f"({result.source_type}) "
            f"baseline={result.baseline_tokens} "
            f"optimized={result.optimized_tokens} "
            f"saved={result.saved_tokens} "
            f"ratio={result.saved_ratio:.4f}"
        )
        if result.failure_reasons:
            for reason in result.failure_reasons:
                lines.append(f"         - {reason}")
    return "\n".join(lines)


def _run_tool_schema_fixture(token_counter: TokenCounter) -> ToolSchemaOptimizationOutcome:
    catalog = {
        "tools": [
            {
                "name": "search_files",
                "description": "  Search   the   workspace  ",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "  Search   query  ",
                            "enum": ["name", "content", "path"],
                        }
                    },
                    "required": ["query"],
                },
            }
        ]
    }
    pretty_input = json.dumps(catalog, indent=2)
    return optimize_tool_schema_catalog(
        pretty_input,
        token_policy=DEFAULT_TOOL_CATALOG_TOKEN_POLICY,
        token_counter=token_counter,
    )


def _run_context_pack_fixture(token_counter: TokenCounter) -> ContextPackOptimizationOutcome:
    fragments = [
        ContextFragment(
            fragment_id="evidence_1",
            content="  Retrieved   evidence   fragment  ",
            required=False,
            metadata={"source": "rag", "rank": 1},
        ),
        ContextFragment(
            fragment_id="policy_1",
            content="  Mandatory   policy   text  ",
            required=True,
            metadata={"source": "policy"},
        ),
    ]
    return optimize_context_pack(
        fragments,
        token_policy=DEFAULT_CONTEXT_PACK_TOKEN_POLICY,
        token_counter=token_counter,
    )


def _run_memory_summary_fixture(token_counter: TokenCounter) -> MemorySummaryCompressionOutcome:
    summary = (
        "  Session   summary   for   user   preferences.\n\n\n"
        "User   prefers   concise   answers.\n\n\n\n"
        "Next   step:   review   docs.  "
    )
    return optimize_memory_summary(
        summary,
        token_policy=DEFAULT_MEMORY_SUMMARY_TOKEN_POLICY,
        token_counter=token_counter,
    )


def _resolve_token_metrics(
    outcome: object,
    token_counter: TokenCounter,
) -> tuple[int, int, int, float]:
    original_tokens = getattr(outcome, "original_tokens", None)
    optimized_tokens = getattr(outcome, "optimized_tokens", None)
    saved_tokens = getattr(outcome, "saved_tokens", None)
    saved_ratio = getattr(outcome, "saved_ratio", None)
    if (
        isinstance(original_tokens, int)
        and isinstance(optimized_tokens, int)
        and isinstance(saved_tokens, int)
        and isinstance(saved_ratio, float)
    ):
        return original_tokens, optimized_tokens, saved_tokens, saved_ratio

    result = getattr(outcome, "result", None)
    measurement = getattr(result, "measurement", None) if result is not None else None
    if measurement is not None:
        return (
            measurement.baseline_tokens,
            measurement.optimized_tokens,
            measurement.saved_tokens,
            measurement.saved_ratio,
        )

    original_content = getattr(outcome, "original_content", "")
    optimized_content = getattr(outcome, "optimized_content", "")
    if not isinstance(original_content, str):
        original_content = ""
    if not isinstance(optimized_content, str):
        optimized_content = ""
    baseline = token_counter(original_content)
    optimized = token_counter(optimized_content)
    saved = baseline - optimized
    ratio = saved / baseline if baseline > 0 else 0.0
    return baseline, optimized, saved, ratio


def _resolve_validation_status(outcome: object) -> str:
    direct = getattr(outcome, "validation_status", None)
    if isinstance(direct, ProtectedRegionValidationStatus):
        return direct.value
    if isinstance(direct, str):
        return direct

    protected = getattr(outcome, "protected_region_validation", None)
    if protected is not None and hasattr(protected, "status"):
        status = protected.status
        if isinstance(status, ProtectedRegionValidationStatus):
            return status.value
        if isinstance(status, str):
            return status

    result = getattr(outcome, "result", None)
    validation = getattr(result, "validation", None) if result is not None else None
    if validation is not None and hasattr(validation, "status"):
        status = validation.status
        if isinstance(status, ProtectedRegionValidationStatus):
            return status.value
        if isinstance(status, str):
            return status
    return "unknown"


def _resolve_fallback_status(outcome: object) -> bool:
    direct = getattr(outcome, "fallback_status", None)
    if isinstance(direct, bool):
        return direct
    result = getattr(outcome, "result", None)
    if result is not None:
        fallback_used = getattr(result, "fallback_used", None)
        if isinstance(fallback_used, bool):
            return fallback_used
    return False


def _resolve_receipt_present(outcome: object) -> bool:
    if getattr(outcome, "receipt", None) is not None:
        return True
    if getattr(outcome, "receipt_ref", None) is not None:
        return True
    result = getattr(outcome, "result", None)
    if result is not None and getattr(result, "receipt_ref", None) is not None:
        return True
    return False


def _resolve_strategy_id(outcome: object) -> str | None:
    direct = getattr(outcome, "strategy", None)
    if isinstance(direct, TokenOptimizationStrategyRef):
        return direct.strategy_id

    result = getattr(outcome, "result", None)
    if result is not None:
        strategy = getattr(result, "strategy", None)
        if isinstance(strategy, TokenOptimizationStrategyRef):
            return strategy.strategy_id
    return None


def _evaluate_expectations(
    *,
    expectation: TokenRegressionExpectation,
    baseline_tokens: int,
    optimized_tokens: int,
    saved_tokens: int,
    saved_ratio: float,
    validation_status: str,
    fallback_status: bool,
    receipt_present: bool,
) -> tuple[str, ...]:
    failures: list[str] = []

    savings_expected = (
        (expectation.expected_min_saved_tokens is not None and expectation.expected_min_saved_tokens > 0)
        or (expectation.expected_min_saved_ratio is not None and expectation.expected_min_saved_ratio > 0)
    )
    if savings_expected and optimized_tokens > baseline_tokens:
        failures.append(
            f"optimized_tokens ({optimized_tokens}) exceeded baseline_tokens ({baseline_tokens})"
        )

    if (
        expectation.expected_min_saved_tokens is not None
        and saved_tokens < expectation.expected_min_saved_tokens
    ):
        failures.append(
            "saved_tokens "
            f"({saved_tokens}) below expected_min_saved_tokens "
            f"({expectation.expected_min_saved_tokens})"
        )

    if (
        expectation.expected_min_saved_ratio is not None
        and saved_ratio < expectation.expected_min_saved_ratio
    ):
        failures.append(
            "saved_ratio "
            f"({saved_ratio:.4f}) below expected_min_saved_ratio "
            f"({expectation.expected_min_saved_ratio:.4f})"
        )

    if expectation.require_receipt and not receipt_present:
        failures.append("required receipt missing")

    if expectation.expect_validation_pass and validation_status == ProtectedRegionValidationStatus.FAILED.value:
        failures.append(f"validation failed (status={validation_status})")

    if not expectation.allow_fallback and fallback_status:
        failures.append("fallback used but allow_fallback=False")

    return tuple(failures)


def _build_summary(
    results: Sequence[TokenRegressionResult],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> TokenRegressionSummary:
    total_baseline = sum(result.baseline_tokens for result in results)
    total_optimized = sum(result.optimized_tokens for result in results)
    total_saved = total_baseline - total_optimized
    total_ratio = total_saved / total_baseline if total_baseline > 0 else 0.0
    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    return TokenRegressionSummary(
        total_fixtures=len(results),
        passed=passed,
        failed=failed,
        total_baseline_tokens=total_baseline,
        total_optimized_tokens=total_optimized,
        total_saved_tokens=total_saved,
        total_saved_ratio=total_ratio,
        results=tuple(results),
        metadata=dict(metadata or {}),
    )
