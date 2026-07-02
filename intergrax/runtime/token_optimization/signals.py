# © Artur Czarnecki. All rights reserved.

"""Token optimization domain signal model and safe in-memory emission (Phase TOKEN-OBS-1A)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from intergrax.runtime.token_optimization.context_pack import ContextPackOptimizationOutcome
from intergrax.runtime.token_optimization.contracts import (
    CompressionReceiptRef,
    ProtectedRegionValidationStatus,
    TokenCategory,
    TokenOptimizationAttribution,
    TokenOptimizationResult,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyRef,
)
from intergrax.runtime.token_optimization.receipts import CompressionReceipt
from intergrax.runtime.token_optimization.tool_schema import ToolSchemaOptimizationOutcome

if TYPE_CHECKING:
    from intergrax.runtime.token_optimization.regression import (
        TokenRegressionResult,
        TokenRegressionSummary,
    )

_SIGNAL_ID_PREFIX = "signal_"
_SIGNAL_ID_HASH_LENGTH = 16
_MAX_METADATA_STRING_LENGTH = 160


class TokenOptimizationSignalType(StrEnum):
    """Stable signal type strings for token optimization domain signals."""

    OPTIMIZATION_OUTCOME = "token_optimization.outcome"
    REGRESSION_RESULT = "token_optimization.regression_result"
    REGRESSION_SUMMARY = "token_optimization.regression_summary"


_UNSAFE_METADATA_KEYS: frozenset[str] = frozenset(
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
    }
)

_SAFE_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "run_id",
        "step_id",
        "tenant_id",
        "fixture_id",
        "source_type",
        "token_category",
        "strategy_id",
        "profile",
        "compression_level",
        "validation_status",
        "fallback_status",
        "receipt_id",
        "category",
        "description",
        "mode",
        "input_kind",
        "changed",
        "lines_trimmed",
        "whitespace_compacted",
        "chars_truncated",
        "lossy_truncation_skipped",
        "workflow_id",
        "agent_id",
        "plugin_id",
        "optimization_profile",
        "runtime_profile",
        "token_counter",
        "passed",
        "failure_reasons",
        "total_fixtures",
        "passed_count",
        "failed_count",
    }
)


@dataclass(frozen=True, slots=True)
class TokenOptimizationSignal:
    """Redaction-safe token optimization domain signal for future HOS/export wiring."""

    signal_id: str
    signal_type: TokenOptimizationSignalType
    source_type: TokenOptimizationSourceType | None = None
    token_category: TokenCategory | None = None
    strategy_id: str | None = None
    baseline_tokens: int | None = None
    optimized_tokens: int | None = None
    saved_tokens: int | None = None
    saved_ratio: float | None = None
    validation_status: str | None = None
    fallback_status: bool | None = None
    receipt_id: str | None = None
    receipt_ref: CompressionReceiptRef | None = None
    run_id: str | None = None
    step_id: str | None = None
    tenant_id: str | None = None
    fixture_id: str | None = None
    created_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenOptimizationSignalEmissionResult:
    """Outcome of emitting a token optimization domain signal into a sink."""

    signal: TokenOptimizationSignal
    accepted: bool = True


class TokenOptimizationSignalSink(Protocol):
    """Sink contract for helper-only token optimization domain signal emission."""

    def emit(self, signal: TokenOptimizationSignal) -> None:
        """Accept a token optimization domain signal."""


@dataclass
class InMemoryTokenOptimizationSignalSink:
    """In-memory sink that stores emitted signals for tests."""

    signals: list[TokenOptimizationSignal] = field(default_factory=list)

    def emit(self, signal: TokenOptimizationSignal) -> None:
        self.signals.append(signal)

    def clear(self) -> None:
        self.signals.clear()


@dataclass(frozen=True, slots=True)
class NoOpTokenOptimizationSignalSink:
    """Sink that accepts signals without storing them."""

    def emit(self, signal: TokenOptimizationSignal) -> None:
        return None


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


def _limit_string(value: str, *, max_length: int = _MAX_METADATA_STRING_LENGTH) -> str:
    if len(value) <= max_length:
        return value
    if max_length <= 3:
        return value[:max_length]
    return value[: max_length - 3] + "..."


def sanitize_signal_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a redaction-safe metadata mapping for token optimization signals."""
    if not metadata:
        return {}

    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in _UNSAFE_METADATA_KEYS:
            continue
        if key not in _SAFE_METADATA_KEYS:
            continue
        if isinstance(value, (dict, list, tuple, set)):
            continue
        scalar = _scalar_value(value)
        if scalar is None and value is not None:
            continue
        if isinstance(scalar, str):
            if "\n" in scalar or "\r" in scalar:
                continue
            scalar = _limit_string(scalar)
        sanitized[key] = scalar
    return sanitized


def _scalar_attr_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if "\n" in value or "\r" in value:
        return None
    return _limit_string(value)


def _sanitize_receipt_ref(
    receipt_ref: CompressionReceiptRef | None,
) -> CompressionReceiptRef | None:
    if receipt_ref is None:
        return None
    return CompressionReceiptRef(
        receipt_id=receipt_ref.receipt_id,
        run_id=receipt_ref.run_id,
        step_id=receipt_ref.step_id,
        strategy_id=receipt_ref.strategy_id,
        original_hash=receipt_ref.original_hash,
        optimized_hash=receipt_ref.optimized_hash,
        metadata=sanitize_signal_metadata(receipt_ref.metadata),
    )


def _derive_signal_id(*parts: str) -> str:
    payload = "|".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_SIGNAL_ID_HASH_LENGTH]
    return f"{_SIGNAL_ID_PREFIX}{digest}"


def _resolve_validation_status_value(value: Any) -> str | None:
    if isinstance(value, ProtectedRegionValidationStatus):
        return value.value
    if isinstance(value, str):
        return value
    return None


def _resolve_source_type_value(value: Any) -> TokenOptimizationSourceType | None:
    if isinstance(value, TokenOptimizationSourceType):
        return value
    if isinstance(value, str):
        try:
            return TokenOptimizationSourceType(value)
        except ValueError:
            return None
    return None


def _resolve_token_category_value(value: Any) -> TokenCategory | None:
    if isinstance(value, TokenCategory):
        return value
    if isinstance(value, str):
        try:
            return TokenCategory(value)
        except ValueError:
            return None
    return None


def _resolve_strategy_id(strategy: TokenOptimizationStrategyRef | None) -> str | None:
    if strategy is None:
        return None
    return strategy.strategy_id


def _resolve_receipt_fields(
    *,
    receipt: CompressionReceipt | None,
    receipt_ref: CompressionReceiptRef | None,
) -> tuple[str | None, CompressionReceiptRef | None, str | None, str | None, str | None]:
    resolved_ref = _sanitize_receipt_ref(receipt_ref)
    receipt_id: str | None = None
    run_id: str | None = None
    step_id: str | None = None
    tenant_id: str | None = None

    if receipt is not None:
        receipt_id = receipt.receipt_id or None
        run_id = receipt.run_id
        step_id = receipt.step_id
        if receipt.attribution is not None:
            tenant_id = receipt.attribution.tenant_id
        if resolved_ref is None and receipt_id:
            resolved_ref = _sanitize_receipt_ref(
                CompressionReceiptRef(
                    receipt_id=receipt_id,
                    run_id=run_id,
                    step_id=step_id,
                    strategy_id=(
                        receipt.strategy.strategy_id if receipt.strategy is not None else None
                    ),
                    original_hash=receipt.original_hash,
                    optimized_hash=receipt.optimized_hash,
                    metadata=receipt.metadata,
                )
            )

    if resolved_ref is not None:
        receipt_id = receipt_id or resolved_ref.receipt_id
        run_id = run_id or resolved_ref.run_id
        step_id = step_id or resolved_ref.step_id

    return receipt_id, _sanitize_receipt_ref(resolved_ref), run_id, step_id, tenant_id


def _resolve_attribution_fields(
    attribution: TokenOptimizationAttribution | None,
    *,
    run_id: str | None,
    step_id: str | None,
    tenant_id: str | None,
) -> tuple[str | None, str | None, str | None]:
    if attribution is None:
        return run_id, step_id, tenant_id
    return (
        run_id or attribution.run_id,
        step_id or attribution.step_id,
        tenant_id or attribution.tenant_id,
    )


def _merge_metadata(*sources: Mapping[str, Any] | None) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for source in sources:
        if source:
            combined.update(source)
    return sanitize_signal_metadata(combined)


def _build_from_result(
    result: TokenOptimizationResult,
    *,
    source_type: TokenOptimizationSourceType | None,
    attribution: TokenOptimizationAttribution | None,
    receipt: CompressionReceipt | None,
    receipt_ref: CompressionReceiptRef | None,
    strategy: TokenOptimizationStrategyRef | None,
    metadata: Mapping[str, Any] | None,
    signal_id: str | None,
    created_at: str | None,
    fixture_id: str | None,
) -> TokenOptimizationSignal:
    measurement = result.measurement
    validation = result.validation

    baseline_tokens: int | None = None
    optimized_tokens: int | None = None
    saved_tokens: int | None = None
    saved_ratio: float | None = None
    token_category: TokenCategory | None = None

    if measurement is not None:
        baseline_tokens = measurement.baseline_tokens
        optimized_tokens = measurement.optimized_tokens
        saved_tokens = measurement.saved_tokens
        saved_ratio = measurement.saved_ratio
        token_category = measurement.category
        if source_type is None:
            source_type = measurement.source_type
        if attribution is None:
            attribution = measurement.attribution

    resolved_strategy = strategy or result.strategy
    strategy_id = _resolve_strategy_id(resolved_strategy)
    if strategy_id is None and attribution is not None:
        strategy_id = attribution.strategy_id
    if token_category is None and attribution is not None:
        token_category = attribution.token_category
    if source_type is None and attribution is not None:
        source_type = attribution.source_type

    receipt_id, resolved_receipt_ref, run_id, step_id, tenant_id = _resolve_receipt_fields(
        receipt=receipt,
        receipt_ref=receipt_ref or result.receipt_ref,
    )
    run_id, step_id, tenant_id = _resolve_attribution_fields(
        attribution,
        run_id=run_id,
        step_id=step_id,
        tenant_id=tenant_id,
    )

    validation_status = _resolve_validation_status_value(
        validation.status if validation is not None else None
    )
    resolved_signal_id = signal_id or _derive_signal_id(
        TokenOptimizationSignalType.OPTIMIZATION_OUTCOME.value,
        strategy_id or "unknown",
        str(baseline_tokens),
        str(optimized_tokens),
        receipt_id or "",
    )
    resolved_created_at = created_at or datetime.now(UTC).isoformat()

    return TokenOptimizationSignal(
        signal_id=resolved_signal_id,
        signal_type=TokenOptimizationSignalType.OPTIMIZATION_OUTCOME,
        source_type=source_type,
        token_category=token_category,
        strategy_id=strategy_id,
        baseline_tokens=baseline_tokens,
        optimized_tokens=optimized_tokens,
        saved_tokens=saved_tokens,
        saved_ratio=saved_ratio,
        validation_status=validation_status,
        fallback_status=result.fallback_used,
        receipt_id=receipt_id,
        receipt_ref=resolved_receipt_ref,
        run_id=run_id,
        step_id=step_id,
        tenant_id=tenant_id,
        fixture_id=fixture_id,
        created_at=resolved_created_at,
        metadata=_merge_metadata(result.metadata, metadata),
    )


def _is_wrapped_optimization_outcome(outcome: object) -> bool:
    return (
        hasattr(outcome, "request")
        and hasattr(outcome, "result")
        and not isinstance(outcome, TokenOptimizationResult)
    )


def build_token_optimization_signal(
    outcome: TokenOptimizationResult | ToolSchemaOptimizationOutcome | ContextPackOptimizationOutcome | Any,
    *,
    attribution: TokenOptimizationAttribution | None = None,
    metadata: Mapping[str, Any] | None = None,
    signal_id: str | None = None,
    created_at: str | None = None,
    fixture_id: str | None = None,
) -> TokenOptimizationSignal:
    """Build one redaction-safe domain signal from an optimization outcome or result."""
    if isinstance(outcome, TokenOptimizationResult):
        return _build_from_result(
            outcome,
            source_type=None,
            attribution=attribution,
            receipt=None,
            receipt_ref=outcome.receipt_ref,
            strategy=outcome.strategy,
            metadata=metadata,
            signal_id=signal_id,
            created_at=created_at,
            fixture_id=fixture_id,
        )

    if _is_wrapped_optimization_outcome(outcome):
        request = outcome.request
        result = outcome.result
        resolved_attribution = attribution or request.attribution
        source_type = getattr(outcome, "source_type", None) or request.source_type
        strategy = getattr(outcome, "strategy", None) or result.strategy

        direct_baseline = getattr(outcome, "original_tokens", None)
        direct_optimized = getattr(outcome, "optimized_tokens", None)
        direct_saved = getattr(outcome, "saved_tokens", None)
        direct_ratio = getattr(outcome, "saved_ratio", None)
        direct_validation = getattr(outcome, "validation_status", None)
        direct_fallback = getattr(outcome, "fallback_status", None)

        signal = _build_from_result(
            result,
            source_type=source_type,
            attribution=resolved_attribution,
            receipt=getattr(outcome, "receipt", None),
            receipt_ref=getattr(outcome, "receipt_ref", None),
            strategy=strategy,
            metadata=_merge_metadata(outcome.metadata, metadata),
            signal_id=signal_id,
            created_at=created_at,
            fixture_id=fixture_id,
        )

        if any(value is not None for value in (direct_baseline, direct_optimized, direct_saved, direct_ratio)):
            signal = TokenOptimizationSignal(
                signal_id=signal.signal_id,
                signal_type=signal.signal_type,
                source_type=signal.source_type,
                token_category=signal.token_category,
                strategy_id=signal.strategy_id,
                baseline_tokens=direct_baseline if isinstance(direct_baseline, int) else signal.baseline_tokens,
                optimized_tokens=(
                    direct_optimized if isinstance(direct_optimized, int) else signal.optimized_tokens
                ),
                saved_tokens=direct_saved if isinstance(direct_saved, int) else signal.saved_tokens,
                saved_ratio=direct_ratio if isinstance(direct_ratio, float) else signal.saved_ratio,
                validation_status=_resolve_validation_status_value(direct_validation) or signal.validation_status,
                fallback_status=(
                    direct_fallback if isinstance(direct_fallback, bool) else signal.fallback_status
                ),
                receipt_id=signal.receipt_id,
                receipt_ref=signal.receipt_ref,
                run_id=signal.run_id,
                step_id=signal.step_id,
                tenant_id=signal.tenant_id,
                fixture_id=signal.fixture_id,
                created_at=signal.created_at,
                metadata=signal.metadata,
            )
        return signal

    raise TypeError(f"unsupported optimization outcome type: {type(outcome)!r}")


def build_token_regression_signal(
    result: TokenRegressionResult,
    *,
    metadata: Mapping[str, Any] | None = None,
    signal_id: str | None = None,
    created_at: str | None = None,
) -> TokenOptimizationSignal:
    """Build one redaction-safe domain signal from a regression benchmark result."""
    combined_metadata = dict(result.metadata)
    if metadata:
        combined_metadata.update(metadata)

    run_id = combined_metadata.get("run_id")
    step_id = combined_metadata.get("step_id")
    tenant_id = combined_metadata.get("tenant_id")

    resolved_signal_id = signal_id or _derive_signal_id(
        TokenOptimizationSignalType.REGRESSION_RESULT.value,
        result.fixture_id,
        result.source_type,
        str(result.baseline_tokens),
        str(result.optimized_tokens),
    )

    return TokenOptimizationSignal(
        signal_id=resolved_signal_id,
        signal_type=TokenOptimizationSignalType.REGRESSION_RESULT,
        source_type=_resolve_source_type_value(result.source_type),
        strategy_id=result.strategy,
        baseline_tokens=result.baseline_tokens,
        optimized_tokens=result.optimized_tokens,
        saved_tokens=result.saved_tokens,
        saved_ratio=result.saved_ratio,
        validation_status=result.validation_status,
        fallback_status=result.fallback_status,
        receipt_id=None,
        receipt_ref=None,
        run_id=_scalar_attr_string(run_id),
        step_id=_scalar_attr_string(step_id),
        tenant_id=_scalar_attr_string(tenant_id),
        fixture_id=result.fixture_id,
        created_at=created_at or datetime.now(UTC).isoformat(),
        metadata=_merge_metadata(
            combined_metadata,
            {
                "passed": result.passed,
                "category": combined_metadata.get("category"),
                "description": combined_metadata.get("description"),
            },
        ),
    )


def build_token_regression_summary_signal(
    summary: TokenRegressionSummary,
    *,
    metadata: Mapping[str, Any] | None = None,
    signal_id: str | None = None,
    created_at: str | None = None,
) -> TokenOptimizationSignal:
    """Build one aggregate redaction-safe domain signal from a regression benchmark summary."""
    combined_metadata = dict(summary.metadata)
    if metadata:
        combined_metadata.update(metadata)

    resolved_signal_id = signal_id or _derive_signal_id(
        TokenOptimizationSignalType.REGRESSION_SUMMARY.value,
        str(summary.total_fixtures),
        str(summary.passed),
        str(summary.failed),
        str(summary.total_baseline_tokens),
    )

    return TokenOptimizationSignal(
        signal_id=resolved_signal_id,
        signal_type=TokenOptimizationSignalType.REGRESSION_SUMMARY,
        baseline_tokens=summary.total_baseline_tokens,
        optimized_tokens=summary.total_optimized_tokens,
        saved_tokens=summary.total_saved_tokens,
        saved_ratio=summary.total_saved_ratio,
        fixture_id=None,
        created_at=created_at or datetime.now(UTC).isoformat(),
        metadata=_merge_metadata(
            combined_metadata,
            {
                "total_fixtures": summary.total_fixtures,
                "passed_count": summary.passed,
                "failed_count": summary.failed,
            },
        ),
    )


def emit_token_optimization_signal(
    signal: TokenOptimizationSignal,
    sink: TokenOptimizationSignalSink,
) -> TokenOptimizationSignalEmissionResult:
    """Emit a token optimization domain signal into the provided sink."""
    sink.emit(signal)
    return TokenOptimizationSignalEmissionResult(signal=signal, accepted=True)
