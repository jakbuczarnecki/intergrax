# © Artur Czarnecki. All rights reserved.

"""Explicit opt-in token optimization emission helpers (Phase TOKEN-OBS-1C)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class TokenOptimizationEmissionResult:
    """Outcome of an explicit opt-in token optimization domain-signal emission."""

    signal: TokenOptimizationSignal
    payload: TokenOptimizationSignalPayloadV1
    event: RuntimeEvent | None
    emitted: bool
    metadata: Mapping[str, Any]


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
