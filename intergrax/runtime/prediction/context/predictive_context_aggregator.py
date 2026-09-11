# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Deterministic predictive context aggregation (PREDICTIVE R4)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from intergrax.contracts.predictive import (
    PREDICTIVE_CONTEXT_VERSION,
    PredictiveContext,
    PredictiveContextCompleteness,
    PredictiveContextDiagnostic,
    PredictiveContextHistory,
    PredictiveContextMetadata,
    PredictiveContextPerformance,
    PredictiveContextProvider,
    PredictiveContextProviderFragment,
    PredictiveContextProviderStatus,
    PredictiveScope,
)
from intergrax.contracts.predictive.provenance import PredictiveContextProvenance


def _concat(*parts: tuple) -> tuple:
    merged: list = []
    for part in parts:
        merged.extend(part)
    return tuple(merged)


def _merge_history(
    fragments: tuple[PredictiveContextProviderFragment, ...],
) -> PredictiveContextHistory:
    return PredictiveContextHistory(
        execution_patterns=_concat(
            *(f.history.execution_patterns for f in fragments if f.history is not None)
        ),
        failure_patterns=_concat(
            *(f.history.failure_patterns for f in fragments if f.history is not None)
        ),
        retry_patterns=_concat(
            *(f.history.retry_patterns for f in fragments if f.history is not None)
        ),
        latency_patterns=_concat(
            *(f.history.latency_patterns for f in fragments if f.history is not None)
        ),
    )


def _merge_performance(
    fragments: tuple[PredictiveContextProviderFragment, ...],
) -> PredictiveContextPerformance:
    return PredictiveContextPerformance(
        latency_series=_concat(
            *(f.performance.latency_series for f in fragments if f.performance is not None)
        ),
        throughput_series=_concat(
            *(f.performance.throughput_series for f in fragments if f.performance is not None)
        ),
        resource_signals=_concat(
            *(f.performance.resource_signals for f in fragments if f.performance is not None)
        ),
    )


def _merge_diagnostic(
    fragments: tuple[PredictiveContextProviderFragment, ...],
) -> PredictiveContextDiagnostic:
    return PredictiveContextDiagnostic(
        previous_findings=_concat(
            *(f.diagnostic.previous_findings for f in fragments if f.diagnostic is not None)
        ),
        previous_risk_signals=_concat(
            *(f.diagnostic.previous_risk_signals for f in fragments if f.diagnostic is not None)
        ),
        historical_problems=_concat(
            *(f.diagnostic.historical_problems for f in fragments if f.diagnostic is not None)
        ),
    )


def _derive_completeness(
    *,
    provider_count: int,
    missing: tuple[str, ...],
    history: PredictiveContextHistory,
    performance: PredictiveContextPerformance,
    diagnostic: PredictiveContextDiagnostic,
) -> PredictiveContextCompleteness:
    if provider_count == 0:
        return PredictiveContextCompleteness.UNAVAILABLE
    if len(missing) == provider_count:
        return PredictiveContextCompleteness.UNAVAILABLE

    has_history = any(
        (
            history.execution_patterns,
            history.failure_patterns,
            history.retry_patterns,
            history.latency_patterns,
        )
    )
    has_performance = any(
        (
            performance.latency_series,
            performance.throughput_series,
            performance.resource_signals,
        )
    )
    has_diagnostic = any(
        (
            diagnostic.previous_findings,
            diagnostic.previous_risk_signals,
            diagnostic.historical_problems,
        )
    )
    if not has_history and not has_performance and not has_diagnostic:
        if missing:
            return PredictiveContextCompleteness.LIMITED
        return PredictiveContextCompleteness.LIMITED
    if missing:
        return PredictiveContextCompleteness.PARTIAL
    return PredictiveContextCompleteness.COMPLETE


def _snapshot_id(scope: PredictiveScope, generated_at: datetime) -> str:
    seed = "|".join(
        (
            scope.tenant_id,
            scope.task_id or "",
            scope.run_id or "",
            scope.execution_id or "",
            generated_at.isoformat(),
        )
    )
    return f"pctx_{sha256(seed.encode()).hexdigest()[:24]}"


def _context_snapshot_id(scope: PredictiveScope, generated_at: datetime) -> str:
    seed = "|".join(
        (
            "csnap",
            scope.tenant_id,
            scope.task_id or "",
            scope.run_id or "",
            scope.execution_id or "",
            generated_at.isoformat(),
        )
    )
    return f"csnap_{sha256(seed.encode()).hexdigest()[:24]}"


@dataclass(frozen=True, slots=True)
class PredictiveContextAggregator:
    """Merge provider fragments with deterministic ordering and fault isolation."""

    providers: tuple[PredictiveContextProvider, ...]
    provider_timeout_ms: int = 200

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.providers, key=lambda p: p.provider_id))
        object.__setattr__(self, "providers", ordered)

    def aggregate(self, scope: PredictiveScope) -> PredictiveContext:
        generated_at = datetime.now(tz=UTC)
        fragments: list[PredictiveContextProviderFragment] = []
        missing: list[str] = []
        provenance_rows: list[PredictiveContextProvenance] = []
        current_state: list[str] = []
        decision_history: list[str] = []
        lineage_patterns: list[str] = []

        for provider in self.providers:
            deadline = time.monotonic() + (self.provider_timeout_ms / 1000.0)
            started = time.monotonic()
            try:
                fragment = provider.build(scope)
            except Exception:
                missing.append(provider.provider_id)
                continue
            if time.monotonic() - started > (self.provider_timeout_ms / 1000.0):
                missing.append(provider.provider_id)
                fragments.append(
                    PredictiveContextProviderFragment(
                        provider_id=provider.provider_id,
                        status=PredictiveContextProviderStatus.TIMEOUT,
                    ),
                )
                continue
            if time.monotonic() > deadline:
                missing.append(provider.provider_id)
                fragments.append(
                    PredictiveContextProviderFragment(
                        provider_id=provider.provider_id,
                        status=PredictiveContextProviderStatus.TIMEOUT,
                    ),
                )
                continue
            if fragment.status is not PredictiveContextProviderStatus.SUCCESS:
                missing.append(provider.provider_id)
            fragments.append(fragment)
            if fragment.status is PredictiveContextProviderStatus.SUCCESS:
                prov = fragment.provenance
                if prov is None:
                    version = getattr(provider, "provider_version", "unknown")
                    prov = PredictiveContextProvenance(
                        source=provider.provider_id,
                        version=version,
                        generated_at=generated_at,
                        tenant_scope=scope.tenant_id,
                    )
                provenance_rows.append(prov)
            current_state.extend(fragment.current_state)
            decision_history.extend(fragment.decision_history)
            lineage_patterns.extend(fragment.lineage_patterns)

        success_fragments = tuple(
            f
            for f in fragments
            if f.status is PredictiveContextProviderStatus.SUCCESS
        )
        history = _merge_history(success_fragments)
        performance = _merge_performance(success_fragments)
        diagnostic = _merge_diagnostic(success_fragments)
        completeness = _derive_completeness(
            provider_count=len(self.providers),
            missing=tuple(missing),
            history=history,
            performance=performance,
            diagnostic=diagnostic,
        )
        input_id = _snapshot_id(scope, generated_at)
        metadata = PredictiveContextMetadata(
            generated_at=generated_at,
            context_version=PREDICTIVE_CONTEXT_VERSION,
            completeness=completeness,
            missing_providers=tuple(dict.fromkeys(missing)),
            input_snapshot_id=input_id,
            context_snapshot_id=_context_snapshot_id(scope, generated_at),
            provenance=tuple(provenance_rows),
        )
        return PredictiveContext(
            scope=scope,
            history=history,
            performance=performance,
            diagnostic=diagnostic,
            metadata=metadata,
            current_state=tuple(current_state),
            decision_history=tuple(decision_history),
            lineage_patterns=tuple(lineage_patterns),
        )


__all__ = ["PredictiveContextAggregator"]
