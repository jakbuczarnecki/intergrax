# © Artur Czarnecki. All rights reserved.

"""Explicit opt-in token optimization emission helpers (Phase TOKEN-OBS-1C/1D)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from intergrax.contracts.event_severity import EventSeverity
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.token_optimization.contracts import TokenOptimizationResult
from intergrax.runtime.token_optimization.domain_events import (
    TokenOptimizationSignalPayloadV1,
    emit_token_optimization_domain_signal,
    token_optimization_signal_to_payload,
)
from intergrax.runtime.token_optimization.signals import (
    TokenOptimizationSignal,
    build_token_optimization_signal,
    build_token_regression_signal,
    build_token_regression_summary_signal,
)

if TYPE_CHECKING:
    from intergrax.runtime.token_optimization.regression import (
        TokenRegressionResult,
        TokenRegressionSummary,
    )


class TokenOptimizationEmissionStatus(StrEnum):
    """Policy-gated emission disposition for token optimization domain signals."""

    EMITTED = "emitted"
    SKIPPED_DISABLED = "skipped_disabled"
    SKIPPED_KIND_DISABLED = "skipped_kind_disabled"
    DRY_RUN = "dry_run"


@dataclass(frozen=True, slots=True)
class TokenOptimizationEmissionPolicy:
    """Runtime policy gate for token optimization domain-signal emission."""

    enabled: bool = False
    emit_optimization_outcomes: bool = True
    emit_regression_results: bool = True
    emit_regression_summaries: bool = True
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class TokenOptimizationEmissionResult:
    """Outcome of an explicit opt-in token optimization domain-signal emission."""

    signal: TokenOptimizationSignal
    payload: TokenOptimizationSignalPayloadV1
    event: RuntimeEvent | None
    emitted: bool
    metadata: Mapping[str, Any]
    status: TokenOptimizationEmissionStatus | str = TokenOptimizationEmissionStatus.EMITTED
    skip_reason: str | None = None


def _resolve_policy_gated_emission(
    policy: TokenOptimizationEmissionPolicy | None,
    *,
    kind_enabled: bool,
) -> tuple[bool, TokenOptimizationEmissionStatus, str | None]:
    effective = policy if policy is not None else TokenOptimizationEmissionPolicy()
    if not effective.enabled:
        return (
            False,
            TokenOptimizationEmissionStatus.SKIPPED_DISABLED,
            "token optimization emission policy is disabled",
        )
    if not kind_enabled:
        return (
            False,
            TokenOptimizationEmissionStatus.SKIPPED_KIND_DISABLED,
            "token optimization emission kind is disabled by policy",
        )
    if effective.dry_run:
        return (
            False,
            TokenOptimizationEmissionStatus.DRY_RUN,
            "token optimization emission policy is in dry-run mode",
        )
    return True, TokenOptimizationEmissionStatus.EMITTED, None


def _with_emission_status(
    result: TokenOptimizationEmissionResult,
    *,
    status: TokenOptimizationEmissionStatus,
    skip_reason: str | None,
) -> TokenOptimizationEmissionResult:
    return TokenOptimizationEmissionResult(
        signal=result.signal,
        payload=result.payload,
        event=result.event,
        emitted=result.emitted,
        metadata=result.metadata,
        status=status,
        skip_reason=skip_reason,
    )


def _emit_built_signal(
    ctx: EmitContext,
    signal: TokenOptimizationSignal,
    *,
    emit: bool = True,
    severity: EventSeverity = EventSeverity.INFO,
    agent_id: str | None = None,
    step_id: str | None = None,
    node_id: str | None = None,
) -> TokenOptimizationEmissionResult:
    payload = token_optimization_signal_to_payload(signal)
    if emit:
        event = emit_token_optimization_domain_signal(
            ctx,
            signal,
            severity=severity,
            agent_id=agent_id,
            step_id=step_id,
            node_id=node_id,
        )
        return TokenOptimizationEmissionResult(
            signal=signal,
            payload=payload,
            event=event,
            emitted=True,
            metadata=dict(payload.metadata),
        )
    return TokenOptimizationEmissionResult(
        signal=signal,
        payload=payload,
        event=None,
        emitted=False,
        metadata=dict(payload.metadata),
    )


def emit_token_optimization_outcome(
    ctx: EmitContext,
    outcome: TokenOptimizationResult | Any,
    *,
    metadata: Mapping[str, Any] | None = None,
    signal_id: str | None = None,
    created_at: str | None = None,
    fixture_id: str | None = None,
    severity: EventSeverity = EventSeverity.INFO,
    agent_id: str | None = None,
    step_id: str | None = None,
    node_id: str | None = None,
    emit: bool = True,
) -> TokenOptimizationEmissionResult:
    """Build and optionally emit a domain signal for an optimization outcome or result."""
    signal = build_token_optimization_signal(
        outcome,
        metadata=metadata,
        signal_id=signal_id,
        created_at=created_at,
        fixture_id=fixture_id,
    )
    return _emit_built_signal(
        ctx,
        signal,
        emit=emit,
        severity=severity,
        agent_id=agent_id,
        step_id=step_id,
        node_id=node_id,
    )


def emit_token_regression_result(
    ctx: EmitContext,
    result: TokenRegressionResult,
    *,
    metadata: Mapping[str, Any] | None = None,
    signal_id: str | None = None,
    created_at: str | None = None,
    severity: EventSeverity = EventSeverity.INFO,
    agent_id: str | None = None,
    step_id: str | None = None,
    node_id: str | None = None,
    emit: bool = True,
) -> TokenOptimizationEmissionResult:
    """Build and optionally emit a domain signal for one regression benchmark result."""
    signal = build_token_regression_signal(
        result,
        metadata=metadata,
        signal_id=signal_id,
        created_at=created_at,
    )
    return _emit_built_signal(
        ctx,
        signal,
        emit=emit,
        severity=severity,
        agent_id=agent_id,
        step_id=step_id,
        node_id=node_id,
    )


def emit_token_regression_summary(
    ctx: EmitContext,
    summary: TokenRegressionSummary,
    *,
    metadata: Mapping[str, Any] | None = None,
    signal_id: str | None = None,
    created_at: str | None = None,
    severity: EventSeverity = EventSeverity.INFO,
    agent_id: str | None = None,
    step_id: str | None = None,
    node_id: str | None = None,
    emit: bool = True,
) -> TokenOptimizationEmissionResult:
    """Build and optionally emit an aggregate domain signal for a regression benchmark summary."""
    signal = build_token_regression_summary_signal(
        summary,
        metadata=metadata,
        signal_id=signal_id,
        created_at=created_at,
    )
    return _emit_built_signal(
        ctx,
        signal,
        emit=emit,
        severity=severity,
        agent_id=agent_id,
        step_id=step_id,
        node_id=node_id,
    )


def maybe_emit_token_optimization_outcome(
    ctx: EmitContext,
    outcome: TokenOptimizationResult | Any,
    *,
    policy: TokenOptimizationEmissionPolicy | None = None,
    metadata: Mapping[str, Any] | None = None,
    signal_id: str | None = None,
    created_at: str | None = None,
    fixture_id: str | None = None,
    severity: EventSeverity = EventSeverity.INFO,
    agent_id: str | None = None,
    step_id: str | None = None,
    node_id: str | None = None,
) -> TokenOptimizationEmissionResult:
    """Policy-gated wrapper around explicit optimization-outcome domain-signal emission."""
    effective = policy if policy is not None else TokenOptimizationEmissionPolicy()
    should_emit, status, skip_reason = _resolve_policy_gated_emission(
        effective,
        kind_enabled=effective.emit_optimization_outcomes,
    )
    result = emit_token_optimization_outcome(
        ctx,
        outcome,
        metadata=metadata,
        signal_id=signal_id,
        created_at=created_at,
        fixture_id=fixture_id,
        severity=severity,
        agent_id=agent_id,
        step_id=step_id,
        node_id=node_id,
        emit=should_emit,
    )
    if should_emit:
        return result
    return _with_emission_status(result, status=status, skip_reason=skip_reason)


def maybe_emit_token_regression_result(
    ctx: EmitContext,
    result: TokenRegressionResult,
    *,
    policy: TokenOptimizationEmissionPolicy | None = None,
    metadata: Mapping[str, Any] | None = None,
    signal_id: str | None = None,
    created_at: str | None = None,
    severity: EventSeverity = EventSeverity.INFO,
    agent_id: str | None = None,
    step_id: str | None = None,
    node_id: str | None = None,
) -> TokenOptimizationEmissionResult:
    """Policy-gated wrapper around explicit regression-result domain-signal emission."""
    effective = policy if policy is not None else TokenOptimizationEmissionPolicy()
    should_emit, status, skip_reason = _resolve_policy_gated_emission(
        effective,
        kind_enabled=effective.emit_regression_results,
    )
    emission_result = emit_token_regression_result(
        ctx,
        result,
        metadata=metadata,
        signal_id=signal_id,
        created_at=created_at,
        severity=severity,
        agent_id=agent_id,
        step_id=step_id,
        node_id=node_id,
        emit=should_emit,
    )
    if should_emit:
        return emission_result
    return _with_emission_status(
        emission_result,
        status=status,
        skip_reason=skip_reason,
    )


def maybe_emit_token_regression_summary(
    ctx: EmitContext,
    summary: TokenRegressionSummary,
    *,
    policy: TokenOptimizationEmissionPolicy | None = None,
    metadata: Mapping[str, Any] | None = None,
    signal_id: str | None = None,
    created_at: str | None = None,
    severity: EventSeverity = EventSeverity.INFO,
    agent_id: str | None = None,
    step_id: str | None = None,
    node_id: str | None = None,
) -> TokenOptimizationEmissionResult:
    """Policy-gated wrapper around explicit regression-summary domain-signal emission."""
    effective = policy if policy is not None else TokenOptimizationEmissionPolicy()
    should_emit, status, skip_reason = _resolve_policy_gated_emission(
        effective,
        kind_enabled=effective.emit_regression_summaries,
    )
    emission_result = emit_token_regression_summary(
        ctx,
        summary,
        metadata=metadata,
        signal_id=signal_id,
        created_at=created_at,
        severity=severity,
        agent_id=agent_id,
        step_id=step_id,
        node_id=node_id,
        emit=should_emit,
    )
    if should_emit:
        return emission_result
    return _with_emission_status(
        emission_result,
        status=status,
        skip_reason=skip_reason,
    )
