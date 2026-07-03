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
    CompressionLevel,
    ProtectedRegionValidationStatus,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationStrategyRef,
)
from intergrax.runtime.token_optimization.tool_schema import ToolSchemaOptimizationConfig
from intergrax.runtime.token_optimization.context_pack import ContextPackOptimizationConfig
from intergrax.memory.summary_compressor import MemorySummaryCompressionConfig
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
    expected_max_saved_tokens: int | None = None
    expected_max_saved_ratio: float | None = None
    expected_validation_status: str | None = None
    expect_fallback: bool | None = None
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
        combined_metadata["expectation_status"] = "met" if not failure_reasons else "failed"

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
                expected_min_saved_ratio=0.05,
                require_receipt=True,
                expect_validation_pass=True,
                allow_fallback=False,
                expect_fallback=False,
            ),
            runner=_run_tool_schema_fixture,
            metadata=_eval_fixture_metadata(
                category="tool_catalog",
                eval_case="compactable",
                expected_behavior="saves_tokens_with_receipt",
            ),
        ),
        TokenRegressionFixture(
            fixture_id="tool_schema.protected_description",
            source_type=TokenRegressionSourceType.TOOL_SCHEMA,
            description="Protected URL in tool description is preserved without unsafe compaction.",
            expectation=TokenRegressionExpectation(
                expected_max_saved_tokens=0,
                expected_max_saved_ratio=0.0,
                require_receipt=True,
                expect_validation_pass=True,
                allow_fallback=False,
                expect_fallback=False,
            ),
            runner=_run_tool_schema_protected_fixture,
            metadata=_eval_fixture_metadata(
                category="tool_catalog",
                eval_case="protected",
                expected_behavior="preserves_protected_regions_no_savings",
            ),
        ),
        TokenRegressionFixture(
            fixture_id="context_pack.compact_fragments",
            source_type=TokenRegressionSourceType.CONTEXT_PACK,
            description="Compact context pack fragments with whitespace normalization.",
            expectation=TokenRegressionExpectation(
                expected_min_saved_tokens=1,
                expected_min_saved_ratio=0.05,
                require_receipt=True,
                expect_validation_pass=True,
                allow_fallback=False,
                expect_fallback=False,
            ),
            runner=_run_context_pack_fixture,
            metadata=_eval_fixture_metadata(
                category="rag_context_pack",
                eval_case="compactable",
                expected_behavior="saves_tokens_with_receipt",
            ),
        ),
        TokenRegressionFixture(
            fixture_id="context_pack.protected_evidence",
            source_type=TokenRegressionSourceType.CONTEXT_PACK,
            description="Protected evidence reference fragment is preserved without unsafe compaction.",
            expectation=TokenRegressionExpectation(
                expected_max_saved_tokens=0,
                expected_max_saved_ratio=0.0,
                require_receipt=True,
                expect_validation_pass=True,
                allow_fallback=False,
                expect_fallback=False,
            ),
            runner=_run_context_pack_protected_fixture,
            metadata=_eval_fixture_metadata(
                category="rag_context_pack",
                eval_case="protected",
                expected_behavior="preserves_protected_evidence_no_savings",
            ),
        ),
        TokenRegressionFixture(
            fixture_id="memory_summary.compact_summary",
            source_type=TokenRegressionSourceType.MEMORY_SUMMARY,
            description="Compact memory summary with structural whitespace normalization.",
            expectation=TokenRegressionExpectation(
                expected_min_saved_ratio=0.0,
                require_receipt=True,
                expect_validation_pass=True,
                allow_fallback=False,
                expect_fallback=False,
            ),
            runner=_run_memory_summary_fixture,
            metadata=_eval_fixture_metadata(
                category="memory",
                eval_case="compactable",
                expected_behavior="applies_structural_compaction_with_receipt",
            ),
        ),
        TokenRegressionFixture(
            fixture_id="memory_summary.protected_dates",
            source_type=TokenRegressionSourceType.MEMORY_SUMMARY,
            description="Protected dates in memory summary are preserved without fallback.",
            expectation=TokenRegressionExpectation(
                expected_max_saved_tokens=0,
                expected_max_saved_ratio=0.0,
                require_receipt=True,
                expect_validation_pass=True,
                allow_fallback=False,
                expect_fallback=False,
            ),
            runner=_run_memory_summary_protected_fixture,
            metadata=_eval_fixture_metadata(
                category="memory",
                eval_case="protected",
                expected_behavior="preserves_protected_dates_no_savings",
            ),
        ),
        TokenRegressionFixture(
            fixture_id="memory_summary.fallback_validation",
            source_type=TokenRegressionSourceType.MEMORY_SUMMARY,
            description="Lossy truncation that would break protected dates falls back to original content.",
            expectation=TokenRegressionExpectation(
                expected_max_saved_tokens=0,
                expected_max_saved_ratio=0.0,
                expected_validation_status=ProtectedRegionValidationStatus.FAILED.value,
                expect_fallback=True,
                require_receipt=True,
                expect_validation_pass=False,
                allow_fallback=True,
            ),
            runner=_run_memory_summary_fallback_fixture,
            metadata=_eval_fixture_metadata(
                category="memory",
                eval_case="fallback",
                expected_behavior="falls_back_on_validation_failure",
            ),
        ),
    )


def _eval_fixture_metadata(
    *,
    category: str,
    eval_case: str,
    expected_behavior: str,
) -> dict[str, str]:
    return {
        "category": category,
        "eval_case": eval_case,
        "expected_behavior": expected_behavior,
    }


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


def _run_tool_schema_protected_fixture(
    token_counter: TokenCounter,
) -> ToolSchemaOptimizationOutcome:
    prefix = "context " * 40
    protected_url = "https://example.com/protected/resource"
    description = f"{prefix}See {protected_url} for details and more usage notes."
    catalog = {
        "tools": [
            {
                "name": "search_files",
                "description": description,
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]
    }
    pretty_input = json.dumps(catalog, indent=2)
    return optimize_tool_schema_catalog(
        pretty_input,
        token_policy=DEFAULT_TOOL_CATALOG_TOKEN_POLICY,
        config=ToolSchemaOptimizationConfig(max_description_chars=120),
        token_counter=token_counter,
    )


def _run_context_pack_protected_fixture(
    token_counter: TokenCounter,
) -> ContextPackOptimizationOutcome:
    prefix = "context " * 40
    evidence_ref = "evidence_abcdefgh1234"
    content = f"{prefix}See {evidence_ref} for audit trail and more notes."
    fragments = [
        ContextFragment(
            fragment_id="evidence_frag",
            content=content,
            required=False,
            metadata={"source": "evidence"},
        ),
    ]
    return optimize_context_pack(
        fragments,
        token_policy=DEFAULT_CONTEXT_PACK_TOKEN_POLICY,
        config=ContextPackOptimizationConfig(max_fragment_chars=120),
        token_counter=token_counter,
    )


def _run_memory_summary_protected_fixture(
    token_counter: TokenCounter,
) -> MemorySummaryCompressionOutcome:
    summary = "  Meeting   on   2026-07-01   was   scheduled.\n\n\nFollow   up   later.  "
    return optimize_memory_summary(
        summary,
        token_policy=DEFAULT_MEMORY_SUMMARY_TOKEN_POLICY,
        token_counter=token_counter,
    )


def _run_memory_summary_fallback_fixture(
    token_counter: TokenCounter,
) -> MemorySummaryCompressionOutcome:
    summary = (
        "User prefers concise answers.\n"
        "Follow-up scheduled on 2026-07-01.\n"
        "User does not want runtime memory wiring yet."
    )
    lossy_policy = TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.CONSERVATIVE,
        compression_level=CompressionLevel.LIGHT,
        allow_lossy=True,
        require_validation=True,
        fallback_on_validation_failure=True,
        emit_receipts=True,
    )
    config = MemorySummaryCompressionConfig(
        compact_whitespace=False,
        trim_blank_lines=False,
        trim_edges=False,
        max_summary_chars=55,
    )
    return optimize_memory_summary(
        summary,
        token_policy=lossy_policy,
        config=config,
        semantic_validation_hook=_accept_semantic_validation,
        token_counter=token_counter,
    )


def _accept_semantic_validation(
    _original_content: str,
    _optimized_content: str,
    _metadata: object,
) -> bool:
    return True


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
    return "missing"


_PASS_LIKE_VALIDATION_STATUSES = frozenset(
    {
        ProtectedRegionValidationStatus.PASSED.value,
        ProtectedRegionValidationStatus.NOT_APPLICABLE.value,
    }
)


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

    if (
        expectation.expected_max_saved_tokens is not None
        and saved_tokens > expectation.expected_max_saved_tokens
    ):
        failures.append(
            "saved_tokens "
            f"({saved_tokens}) above expected_max_saved_tokens "
            f"({expectation.expected_max_saved_tokens})"
        )

    if (
        expectation.expected_max_saved_ratio is not None
        and saved_ratio > expectation.expected_max_saved_ratio
    ):
        failures.append(
            "saved_ratio "
            f"({saved_ratio:.4f}) above expected_max_saved_ratio "
            f"({expectation.expected_max_saved_ratio:.4f})"
        )

    if expectation.require_receipt and not receipt_present:
        failures.append("required receipt missing")

    if expectation.expected_validation_status is not None:
        if validation_status != expectation.expected_validation_status:
            failures.append(
                "validation status "
                f"({validation_status}) did not match expected_validation_status "
                f"({expectation.expected_validation_status})"
            )

    if expectation.expect_fallback is True and not fallback_status:
        failures.append("expected fallback but fallback was not used")

    if expectation.expect_fallback is False and fallback_status:
        failures.append("unexpected fallback used")

    if (
        expectation.expect_validation_pass
        and validation_status not in _PASS_LIKE_VALIDATION_STATUSES
    ):
        failures.append(
            f"validation status was not pass-like (status={validation_status})"
        )

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
