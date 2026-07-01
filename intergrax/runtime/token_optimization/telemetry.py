# © Artur Czarnecki. All rights reserved.

"""Token savings telemetry payload contracts and helpers (Phase TOKEN-6A-lite)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegionValidationStatus,
    TokenCategory,
    TokenOptimizationAttribution,
    TokenOptimizationBypassReason,
    TokenOptimizationDecision,
    TokenOptimizationMechanism,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
    TokenSavingsClaimConfidence,
)
from intergrax.runtime.token_optimization.receipts import CompressionReceipt

_ATTRIBUTE_PREFIX = "intergrax.token_optimization."


class TokenOptimizationTelemetryEventType(StrEnum):
    """Stable event type strings for token optimization telemetry."""

    TOKEN_SAVINGS = "token_optimization.savings"


class TokenOptimizationTelemetryValidationStatus(StrEnum):
    """Outcome of deterministic telemetry payload validation."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class TokenOptimizationTelemetryValidationResult:
    """Telemetry payload validation outcome."""

    status: TokenOptimizationTelemetryValidationStatus
    failures: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenOptimizationTelemetryPayload:
    """Typed token-savings telemetry payload shape for future HOS/domain-signal emission."""

    event_type: TokenOptimizationTelemetryEventType
    receipt_id: str
    source_type: TokenOptimizationSourceType
    decision: TokenOptimizationDecision
    run_id: str | None = None
    step_id: str | None = None
    workflow_id: str | None = None
    tenant_id: str | None = None
    agent_id: str | None = None
    model: str | None = None
    provider: str | None = None
    runtime_profile: str | None = None
    optimization_profile: TokenOptimizationProfile | None = None
    token_category: TokenCategory | None = None
    strategy_id: str | None = None
    mechanism: TokenOptimizationMechanism | None = None
    strategy_kind: TokenOptimizationStrategyKind | None = None
    plugin_id: str | None = None
    fallback_used: bool = False
    bypass_reason: TokenOptimizationBypassReason | None = None
    validation_status: ProtectedRegionValidationStatus | None = None
    baseline_tokens: int | None = None
    optimized_tokens: int | None = None
    saved_tokens: int | None = None
    saved_ratio: float | None = None
    savings_confidence: TokenSavingsClaimConfidence | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _resolve_run_id(
    receipt: CompressionReceipt,
    attribution: TokenOptimizationAttribution | None,
) -> str | None:
    if receipt.run_id is not None:
        return receipt.run_id
    if attribution is not None:
        return attribution.run_id
    return None


def _resolve_step_id(
    receipt: CompressionReceipt,
    attribution: TokenOptimizationAttribution | None,
) -> str | None:
    if receipt.step_id is not None:
        return receipt.step_id
    if attribution is not None:
        return attribution.step_id
    return None


def _merge_metadata(
    receipt: CompressionReceipt,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    if receipt.metadata:
        combined.update(receipt.metadata)
    if metadata:
        combined.update(metadata)
    return combined


def build_token_savings_telemetry_payload(
    *,
    receipt: CompressionReceipt,
    workflow_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TokenOptimizationTelemetryPayload:
    """Build a typed token-savings telemetry payload from a compression receipt."""
    measurement = receipt.measurement
    strategy = receipt.strategy
    attribution = receipt.attribution

    resolved_workflow_id = workflow_id
    if resolved_workflow_id is None and attribution is not None:
        resolved_workflow_id = attribution.workflow_id

    token_category: TokenCategory | None = None
    baseline_tokens: int | None = None
    optimized_tokens: int | None = None
    saved_tokens: int | None = None
    saved_ratio: float | None = None
    savings_confidence: TokenSavingsClaimConfidence | None = None

    if measurement is not None:
        token_category = measurement.category
        baseline_tokens = measurement.baseline_tokens
        optimized_tokens = measurement.optimized_tokens
        saved_tokens = measurement.saved_tokens
        saved_ratio = measurement.saved_ratio
        savings_confidence = measurement.confidence

    strategy_id: str | None = None
    mechanism: TokenOptimizationMechanism | None = None
    strategy_kind: TokenOptimizationStrategyKind | None = None
    plugin_id = receipt.plugin_id

    if strategy is not None:
        strategy_id = strategy.strategy_id
        mechanism = strategy.mechanism
        strategy_kind = strategy.kind
        if plugin_id is None:
            plugin_id = strategy.plugin_id

    if strategy_id is None and attribution is not None:
        strategy_id = attribution.strategy_id
    if plugin_id is None and attribution is not None:
        plugin_id = attribution.plugin_id
    if token_category is None and attribution is not None:
        token_category = attribution.token_category

    validation_status: ProtectedRegionValidationStatus | None = None
    if receipt.validation is not None:
        validation_status = receipt.validation.status

    return TokenOptimizationTelemetryPayload(
        event_type=TokenOptimizationTelemetryEventType.TOKEN_SAVINGS,
        receipt_id=receipt.receipt_id,
        run_id=_resolve_run_id(receipt, attribution),
        step_id=_resolve_step_id(receipt, attribution),
        workflow_id=resolved_workflow_id,
        tenant_id=attribution.tenant_id if attribution is not None else None,
        agent_id=attribution.agent_id if attribution is not None else None,
        model=attribution.model if attribution is not None else None,
        provider=attribution.provider if attribution is not None else None,
        runtime_profile=attribution.runtime_profile if attribution is not None else None,
        optimization_profile=(
            attribution.optimization_profile if attribution is not None else None
        ),
        source_type=receipt.source_type,
        token_category=token_category,
        strategy_id=strategy_id,
        mechanism=mechanism,
        strategy_kind=strategy_kind,
        plugin_id=plugin_id,
        decision=receipt.decision,
        fallback_used=receipt.fallback_used,
        bypass_reason=receipt.bypass_reason,
        validation_status=validation_status,
        baseline_tokens=baseline_tokens,
        optimized_tokens=optimized_tokens,
        saved_tokens=saved_tokens,
        saved_ratio=saved_ratio,
        savings_confidence=savings_confidence,
        metadata=_merge_metadata(receipt, metadata),
    )


_FALLBACK_DECISIONS: frozenset[TokenOptimizationDecision] = frozenset(
    {
        TokenOptimizationDecision.FALLBACK,
        TokenOptimizationDecision.BYPASS,
        TokenOptimizationDecision.FAILED,
    }
)


def validate_token_savings_telemetry_payload(
    payload: TokenOptimizationTelemetryPayload,
) -> TokenOptimizationTelemetryValidationResult:
    """Validate a telemetry payload deterministically without exposing raw content."""
    failures: list[str] = []

    if payload.event_type is not TokenOptimizationTelemetryEventType.TOKEN_SAVINGS:
        failures.append("event_type must be token_optimization.savings")

    if not payload.receipt_id:
        failures.append("receipt_id must not be empty")

    if payload.source_type is None:
        failures.append("source_type must be present")

    if payload.decision is None:
        failures.append("decision must be present")

    has_token_fields = (
        payload.baseline_tokens is not None
        or payload.optimized_tokens is not None
        or payload.saved_tokens is not None
        or payload.saved_ratio is not None
    )

    if has_token_fields:
        if payload.baseline_tokens is None:
            failures.append("baseline_tokens must be present when token fields are set")
        elif payload.baseline_tokens < 0:
            failures.append("baseline_tokens must not be negative")

        if payload.optimized_tokens is None:
            failures.append("optimized_tokens must be present when token fields are set")
        elif payload.optimized_tokens < 0:
            failures.append("optimized_tokens must not be negative")

        if (
            payload.baseline_tokens is not None
            and payload.optimized_tokens is not None
            and payload.saved_tokens is not None
        ):
            expected_saved = payload.baseline_tokens - payload.optimized_tokens
            if payload.saved_tokens != expected_saved:
                failures.append("saved_tokens must equal baseline_tokens - optimized_tokens")

        if (
            payload.baseline_tokens is not None
            and payload.baseline_tokens > 0
            and payload.saved_ratio is not None
            and not 0.0 <= payload.saved_ratio <= 1.0
        ):
            failures.append("saved_ratio must be between 0.0 and 1.0 when baseline_tokens > 0")

    if payload.fallback_used and payload.decision not in _FALLBACK_DECISIONS:
        failures.append("decision must be fallback, bypass, or failed when fallback_used is true")

    if failures:
        return TokenOptimizationTelemetryValidationResult(
            status=TokenOptimizationTelemetryValidationStatus.FAILED,
            failures=tuple(failures),
        )

    if not has_token_fields:
        return TokenOptimizationTelemetryValidationResult(
            status=TokenOptimizationTelemetryValidationStatus.NOT_APPLICABLE,
        )

    return TokenOptimizationTelemetryValidationResult(
        status=TokenOptimizationTelemetryValidationStatus.PASSED,
    )


def _scalar_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, StrEnum):
        return value.value
    return None


def token_savings_payload_to_attributes(
    payload: TokenOptimizationTelemetryPayload,
) -> Mapping[str, Any]:
    """Prepare a safe namespaced attribute mapping for future HOS/domain-signal emission."""
    field_map: tuple[tuple[str, Any], ...] = (
        ("event_type", payload.event_type),
        ("receipt_id", payload.receipt_id),
        ("run_id", payload.run_id),
        ("step_id", payload.step_id),
        ("workflow_id", payload.workflow_id),
        ("source_type", payload.source_type),
        ("token_category", payload.token_category),
        ("strategy_id", payload.strategy_id),
        ("plugin_id", payload.plugin_id),
        ("decision", payload.decision),
        ("fallback_used", payload.fallback_used),
        ("validation_status", payload.validation_status),
        ("baseline_tokens", payload.baseline_tokens),
        ("optimized_tokens", payload.optimized_tokens),
        ("saved_tokens", payload.saved_tokens),
        ("saved_ratio", payload.saved_ratio),
        ("savings_confidence", payload.savings_confidence),
    )

    attributes: dict[str, Any] = {}
    for key, raw_value in field_map:
        scalar = _scalar_value(raw_value)
        if scalar is not None or raw_value is None:
            attributes[f"{_ATTRIBUTE_PREFIX}{key}"] = scalar
    return attributes
