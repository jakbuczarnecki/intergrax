# © Artur Czarnecki. All rights reserved.

"""Policy-gated regression benchmark emission wrapper (Phase TOKEN-OBS-1E)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.token_optimization.emission import (
    TokenOptimizationEmissionPolicy,
    TokenOptimizationEmissionResult,
    maybe_emit_token_regression_result,
    maybe_emit_token_regression_summary,
)
from intergrax.runtime.token_optimization.regression import (
    TokenCounter,
    TokenRegressionFixture,
    TokenRegressionSummary,
    default_token_counter,
    run_token_regression_benchmarks,
)


@dataclass(frozen=True, slots=True)
class TokenRegressionEmissionRunResult:
    """Outcome of a regression benchmark run with optional policy-gated emission."""

    summary: TokenRegressionSummary
    result_emissions: tuple[TokenOptimizationEmissionResult, ...]
    summary_emission: TokenOptimizationEmissionResult | None
    emitted_event_count: int
    metadata: Mapping[str, Any]


def run_token_regression_benchmarks_with_emission(
    ctx: EmitContext,
    *,
    fixtures: Sequence[TokenRegressionFixture] | None = None,
    token_counter: TokenCounter = default_token_counter,
    emission_policy: TokenOptimizationEmissionPolicy | None = None,
    emit_results: bool = True,
    emit_summary: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> TokenRegressionEmissionRunResult:
    """Run deterministic regression benchmarks and optionally emit domain signals."""
    summary = run_token_regression_benchmarks(
        token_counter=token_counter,
        fixtures=fixtures,
    )

    effective_metadata = dict(metadata or {})
    result_emissions: list[TokenOptimizationEmissionResult] = []

    if emit_results:
        for result in summary.results:
            emission = maybe_emit_token_regression_result(
                ctx,
                result,
                policy=emission_policy,
                metadata=effective_metadata or None,
            )
            result_emissions.append(emission)

    summary_emission: TokenOptimizationEmissionResult | None = None
    if emit_summary:
        summary_emission = maybe_emit_token_regression_summary(
            ctx,
            summary,
            policy=emission_policy,
            metadata=effective_metadata or None,
        )

    emitted_event_count = sum(1 for emission in result_emissions if emission.emitted)
    if summary_emission is not None and summary_emission.emitted:
        emitted_event_count += 1

    return TokenRegressionEmissionRunResult(
        summary=summary,
        result_emissions=tuple(result_emissions),
        summary_emission=summary_emission,
        emitted_event_count=emitted_event_count,
        metadata=effective_metadata,
    )
