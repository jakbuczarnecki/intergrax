# © Artur Czarnecki. All rights reserved.

"""HOS domain-signal adapter for token optimization signals (Phase TOKEN-OBS-1B)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from intergrax.contracts.event_severity import EventSeverity
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_kind_registry import register_event_kind
from intergrax.runtime.events.payload_registry import register_payload_schema
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.signals import emit_domain_signal
from intergrax.runtime.token_optimization.signals import (
    TokenOptimizationSignal,
    TokenOptimizationSignalType,
    sanitize_signal_metadata,
)

TOKEN_OPTIMIZATION_SIGNAL_EVENT_KIND = "intergrax.token_optimization.signal"
TOKEN_OPTIMIZATION_SIGNAL_PAYLOAD_SCHEMA_ID = "intergrax.token_optimization.signal.v1"


class TokenOptimizationSignalPayloadV1(RuntimeEventPayload):
    """Typed HOS payload for redaction-safe token optimization domain signals."""

    schema_id = TOKEN_OPTIMIZATION_SIGNAL_PAYLOAD_SCHEMA_ID

    signal_id: str
    signal_type: str
    source_type: str | None = None
    token_category: str | None = None
    strategy_id: str | None = None
    baseline_tokens: int | None = None
    optimized_tokens: int | None = None
    saved_tokens: int | None = None
    saved_ratio: float | None = None
    validation_status: str | None = None
    fallback_status: bool | None = None
    receipt_id: str | None = None
    run_id: str | None = None
    step_id: str | None = None
    tenant_id: str | None = None
    fixture_id: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    receipt_run_id: str | None = None
    receipt_step_id: str | None = None
    receipt_strategy_id: str | None = None
    receipt_original_hash: str | None = None
    receipt_optimized_hash: str | None = None
    receipt_metadata: dict[str, Any] = Field(default_factory=dict)

    def redact(self) -> TokenOptimizationSignalPayloadV1:
        """Return a production-safe copy; fields are sanitized at conversion time."""
        return self


def _enum_value(value: StrEnum | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, StrEnum):
        return value.value
    return value


def register_token_optimization_domain_signal() -> None:
    """Register token optimization payload schema and domain event kind (idempotent)."""
    register_payload_schema(TokenOptimizationSignalPayloadV1, extension=True)
    register_event_kind(
        TOKEN_OPTIMIZATION_SIGNAL_EVENT_KIND,
        TOKEN_OPTIMIZATION_SIGNAL_PAYLOAD_SCHEMA_ID,
    )


def token_optimization_signal_to_payload(
    signal: TokenOptimizationSignal,
) -> TokenOptimizationSignalPayloadV1:
    """Convert a token optimization domain signal into a typed HOS payload."""
    receipt_ref = signal.receipt_ref
    receipt_run_id: str | None = None
    receipt_step_id: str | None = None
    receipt_strategy_id: str | None = None
    receipt_original_hash: str | None = None
    receipt_optimized_hash: str | None = None
    receipt_metadata: dict[str, Any] = {}

    if receipt_ref is not None:
        receipt_run_id = receipt_ref.run_id
        receipt_step_id = receipt_ref.step_id
        receipt_strategy_id = receipt_ref.strategy_id
        receipt_original_hash = receipt_ref.original_hash
        receipt_optimized_hash = receipt_ref.optimized_hash
        receipt_metadata = sanitize_signal_metadata(receipt_ref.metadata)

    return TokenOptimizationSignalPayloadV1(
        signal_id=signal.signal_id,
        signal_type=_enum_value(signal.signal_type)
        or TokenOptimizationSignalType.OPTIMIZATION_OUTCOME.value,
        source_type=_enum_value(signal.source_type),
        token_category=_enum_value(signal.token_category),
        strategy_id=signal.strategy_id,
        baseline_tokens=signal.baseline_tokens,
        optimized_tokens=signal.optimized_tokens,
        saved_tokens=signal.saved_tokens,
        saved_ratio=signal.saved_ratio,
        validation_status=signal.validation_status,
        fallback_status=signal.fallback_status,
        receipt_id=signal.receipt_id,
        run_id=signal.run_id,
        step_id=signal.step_id,
        tenant_id=signal.tenant_id,
        fixture_id=signal.fixture_id,
        created_at=signal.created_at,
        metadata=sanitize_signal_metadata(signal.metadata),
        receipt_run_id=receipt_run_id,
        receipt_step_id=receipt_step_id,
        receipt_strategy_id=receipt_strategy_id,
        receipt_original_hash=receipt_original_hash,
        receipt_optimized_hash=receipt_optimized_hash,
        receipt_metadata=receipt_metadata,
    )


def emit_token_optimization_domain_signal(
    ctx: EmitContext,
    signal: TokenOptimizationSignal,
    *,
    severity: EventSeverity = EventSeverity.INFO,
    agent_id: str | None = None,
    step_id: str | None = None,
    node_id: str | None = None,
) -> RuntimeEvent:
    """Emit a token optimization domain signal through the HOS ``emit_domain_signal`` API."""
    register_token_optimization_domain_signal()
    payload = token_optimization_signal_to_payload(signal)
    return emit_domain_signal(
        ctx,
        kind=TOKEN_OPTIMIZATION_SIGNAL_EVENT_KIND,
        payload=payload,
        severity=severity,
        agent_id=agent_id,
        step_id=step_id,
        node_id=node_id,
    )
